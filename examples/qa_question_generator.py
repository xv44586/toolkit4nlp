# -*- coding: utf-8 -*-
# @Date    : 2020/8/20
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : qa_question_generator.py
"""
利用UniLM来通过篇章与答案，来生成问题，可以看做是qa 数据的增强
数据是百度2020icl比赛的机器阅读(http://lic2020.cipsc.org.cn/)
"""
import json
import numpy as np

from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import build_transformer_model,Model
from toolkit4nlp.layers import *
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.utils import pad_sequences, DataGenerator, AutoRegressiveDecoder


# 基本信息
maxlen = 512
epochs = 5
batch_size = 4
learning_rate = 2e-5
max_question_len = 32

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    return D


# 读取数据
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/train.json'
)
val_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/dev.json'
)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    vocab_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
            context, question, answers = item[1:]
            answer = np.random.choice(answers)
            token_ids, _ = tokenizer.encode(answer, context, maxlen=maxlen - max_question_len - 1)
            segment_ids = [0] * len(token_ids)

            question_token_ids = tokenizer.encode(question)[1:]
            token_ids = token_ids + question_token_ids
            segment_ids += [1] * len(question_token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


# loss 层，错位计算预测值并mask掉segment1
class CrossEntropy(Layer):
    def __init__(self, output_axis, *args, **kwargs):
        super(CrossEntropy, self).__init__(self, *args, **kwargs)
        self.output_axis = output_axis

    def call(self, inputs, **kwargs):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:,1:]
        y_pred = y_pred[:, :-1]
        y_mask = y_mask[:, 1:]
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.mean(loss) / K.sum(y_mask)
        self.add_loss(loss)
        return inputs[2]

    def compute_output_shape(self, input_shape):
        return self.input_shape[self.output_axis]

