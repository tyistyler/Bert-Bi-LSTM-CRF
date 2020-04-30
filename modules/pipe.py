
from fastNLP.io import Pipe, ConllLoader
from fastNLP.io import DataBundle
from fastNLP.io.pipe.utils import _add_chars_field, _indexize#给data_bundle中的dataset中复制一列chars. 并根据lower参数判断是否需要小写化
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import Const
from fastNLP.io.utils import check_loader_paths



def bmeso2bio(tags):
    new_tags = []
    for tag in tags:
        tag = tag.lower()
        if tag.startswith('m') or tag.startswith('e'):
            tag = 'i' + tag[1:]
        if tag.startswith('s'):
            tag = 'b' + tag[1:]
        new_tags.append(tag)
    return new_tags

def bmeso2bioes(tags):
    new_tags = []
    for tag in tags:
        lowered_tag = tag.lower()
        if lowered_tag.startswith('m'):
            tag = 'i' + tag[1:]
        new_tags.append(tag)
    return new_tags










class CNNERPipe(Pipe):
    def __init__(self, bigrams=False,encoding_type='bmeso'):
        super().__init__()
        self.bigrams = bigrams
        if encoding_type == 'bmeso':
            self.encoding_func = lambda x:x
        elif encoding_type == 'bio':
            self.encoding_func = bmeso2bio
        elif encoding_type == 'bioes':
            self.encoding_func = bmeso2bioes
        else:
            raise RuntimeError("Only support bio, bmeso, bioes")
    def process(self, data_bundle: DataBundle):
        _add_chars_field(data_bundle, lower=False)
        #field_namae中的标签经过encoding_func函数变化之后，存放在new_field_name中
        data_bundle.apply_field(self.encoding_func, field_name=Const.TARGET, new_field_name=Const.TARGET)
        # 将所有digit转为0
        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                                field_name=Const.CHAR_INPUT, new_field_name=Const.CHAR_INPUT)

        input_field_names = [Const.CHAR_INPUT]
        if self.bigrams:
            data_bundle.apply_field(lambda chars:[c1+c2 for c1,c2 in zip(chars, chars[1:]+['<eos>'])],
                                                        field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')

        # index
        #在dataset中的field_name列建立词表，Const.TARGET列建立词表，并把词表加入到data_bundle中--给char、bigrams、target建立词表
        _indexize(data_bundle, input_field_names=input_field_names, target_field_names=Const.TARGET)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        paths = check_loader_paths(paths)
        loader = ConllLoader(headers=['raw_chars','target'])
        data_bundle = loader.load(paths)
        return self.process(data_bundle)






