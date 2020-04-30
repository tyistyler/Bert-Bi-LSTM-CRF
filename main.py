from fastNLP.embeddings import BertEmbedding
from fastNLP import GradientClipCallback, WarmupCallback, EvaluateCallback
from fastNLP import Trainer, RandomSampler, SpanFPreRecMetric
from modules.pipe import CNNERPipe

from models.MyNER import MyNER
from torch import optim
import argparse
device = 0
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo','resume','msra'])

args = parser.parse_args()

dataset = args.dataset
if dataset == 'weibo':
    # n_heads = 4
    # head_dims = 64
    d_model = 128
    num_layers = 1
    lr = 0.001
    n_epoch = 50
else:
    pass

batch_size = 16
warmup_steps = 0.01
model_type = 'bert'

dropout = 0.5
fc_fropout = 0.4

encoding_type = 'bioes'
name = 'caches/{}_{}_{}.pkl'.format(dataset, model_type, encoding_type)

if dataset == 'weibo':
    paths = {'train':'./data/WeiboNER/BIOES/train.all.bmes',
             'dev':'./data/WeiboNER/BIOES/dev.all.bmes',
             'test':'./data/WeiboNER/BIOES/test.all.bmes'}
    # min_freq = 1#词频数小于这个数量的word将被指向UNK


data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)#导入数据
data_bundle.rename_field('chars','words')#bert自己给word编号

embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='./data/bert-base-chinese',include_cls_sep=False,
                      requires_grad=False, auto_truncate=True)

print('train_dataset:-------------',data_bundle.get_dataset('train'))
print('embed-size',embed.embed_size)



model = MyNER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers, d_model=d_model, fc_dropout=fc_fropout)#定义模型

#定义优化器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)#梯度限制在[-5,5]
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))#每次验证dev之前先验证test

if warmup_steps>0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

save_path = './models'
trainer = Trainer(data_bundle.get_dataset('train'),model=model, optimizer=optimizer, batch_size=batch_size, sampler=RandomSampler(),
                  num_workers=2, n_epochs=n_epoch, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'),encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False, use_tqdm=True,
                  print_every=300, save_path=save_path)

trainer.train(load_best_model=False)
