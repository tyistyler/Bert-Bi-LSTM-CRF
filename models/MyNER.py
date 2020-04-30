
import torch.nn as nn
import torch
import torch.nn.functional as F

from fastNLP.modules import LSTM, ConditionalRandomField, allowed_transitions

class MyNER(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, fc_dropout=0.3):
        '''
        param tag_vocab:  Vocabulary
        param embed: bert-embedding
        param num_layers:
        param dropout:
        param fc_dropout: dropout rate before the fc layer
        '''
        super().__init__()

        self.embed = embed
        embed_size = self.embed.embed_size

        self.in_fc = nn.Linear(embed_size, d_model)

        self.lstm = LSTM(d_model, num_layers=num_layers, hidden_size=100, bidirectional=True, batch_first=True)#d_model=128,num_layers=1
        '''
        batch_first:若为 ``True``, 输入和输出 ``Tensor`` 形状为(batch, seq, feature)
        '''
        self.out_fc = nn.Linear(100*2, len(tag_vocab))
        self.fc_dropout = nn.Dropout(fc_dropout)

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField( len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans )


    def _forward(self, words, target):
        mask = words.ne(0)
        words = self.embed(words)#[16,words,768]

        words = self.in_fc(words)#[16,words,128]

        words, _ = self.lstm(words)#[16,words,128]

        words = self.fc_dropout(words)#[16,words,128]

        words = self.out_fc(words)#[16,words,28]

        logits = F.log_softmax(words, dim=-1)#[16,words,28]

        if target is None:
            paths, _ =self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss':loss}

    def forward(self, words, target):
        return self._forward(words, target)

    def predict(self, words):
        return self._forward(words, target=None)



