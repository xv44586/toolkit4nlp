#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com
"""
利用bert做qa任务baseline
数据是百度2020icl比赛的机器阅读(http://lic2020.cipsc.org.cn/)
原始数据F1: 80.78
增加生成问题后数据F1:
"""
import json, os
import numpy as np
from tqdm import tqdm

from toolkit4nlp.backend import keras, K
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import pad_sequences, DataGenerator
from bert4keras.layers import *

K.clear_session()
# 基本信息
maxlen = 256
epochs = 5
batch_size = 16
learing_rate = 5e-5

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

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def split_context(context, max_context_len):
    for i in range(0, len(context) - max_context_len + 1):
        yield context[i: i + max_context_len]


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
            context, questions, answers = item[1:]
            if type(questions) != list:
                question = questions
            else:
                question = questions[0] if np.random.random()> 0.5 else np.random.choice(questions)

            token_ids, segment_ids = tokenizer.encode(question, context, maxlen=maxlen)
            a = np.random.choice(answers)
            a_token_ids = tokenizer.encode(a)[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = pad_sequences(batch_token_ids)
                    batch_segment_ids = pad_sequences(batch_segment_ids)
                    batch_labels = pad_sequences(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class MaskedSoftmax(Layer):
    """在序列长度那一维进行softmax，并mask掉padding部分
    """

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)


model = build_transformer_model(
    config_path,
    checkpoint_path,
)

inputs = [Input(shape=model.input[0].shape[1:]), Input(shape=model.input[1].shape[1:])]
output = model(inputs)
output = Dropout(0.5)(output)
output = Dense(2)(output)
output = MaskedSoftmax()(output)
output = Permute((2, 1), name='premute')(output)

model = Model(inputs, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(learing_rate),
    metrics=[sparse_accuracy]
)


def extract_answer(question, context, max_a_len=16):
    """抽取答案函数
    """
    max_q_len = 48
    q_token_ids = tokenizer.encode(question, maxlen=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, maxlen=maxlen - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    probas = model.predict([[token_ids], [segment_ids]])[0]
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]


def predict_to_file(infile, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1])
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()


def evaluate(filename):
    """评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json')
    metrics = json.loads(
        os.popen(
            'python /home/mingming.xu/datasets/NLP/qa/dureader_robust-data/evaluate.py %s %s'
            % (filename, filename + '.pred.json')
        ).read().strip()
    )
    return metrics


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(
            '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/dev.json'
        )
        if float(metrics['F1']) >= float(self.best_val_f1):
            self.best_val_f1 = metrics['F1']
            model.save_weights('best_model.weights')
        metrics['BEST F1'] = self.best_val_f1
        print(metrics)


if __name__ == '__main__':
    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('best_model.weights')
