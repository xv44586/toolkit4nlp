# -*- coding: utf-8 -*-
# @Date    : 2020/9/15
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_ifytek_with_similarity.py
"""
利用label description 信息，来增加一个similarity 任务。
具体：对每个样本后面，新增一条由label description 为text， label id为target的新样本，希望每个样本除了学习对应的target外，
样本与 对应label description 样本的similarity 尽可能的高于同一个batch 内的其他样本。

best val acc: 55.4%
"""

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rouge import Rouge

from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.layers import *
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.utils import pad_sequences, DataGenerator, AutoRegressiveDecoder
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.layers import Loss
from keras.callbacks import Callback

num_classes = 119
maxlen = 128
batch_size = 32
max_label = 7
# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], l['label_des']
            D.append((text, int(label), label_des))
    return D


# 加载数据集
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/train.json'
)
valid_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/dev.json'
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

    def __init__(self, sim=False, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        self.sim = sim

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.get_sample(shuffle):
            if not self.sim:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])

            else:
                text_token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
                label_token_ids = tokenizer.encode(label_des, maxlen=max_label + 2)[0][1:]
                token_ids = [text_token_ids] + [label_token_ids]
                segment_ids = [[0] * len(text_token_ids)] + [[0] * len(label_token_ids)]
                batch_token_ids.extend(token_ids)
                batch_segment_ids.extend(segment_ids)
                batch_labels.extend([[label]] * 2)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_sim_generator = data_generator(data=train_data, batch_size=batch_size, sim=True)
train_clf_generator = data_generator(data=train_data, batch_size=batch_size, sim=False)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)


class DenseSimLoss(Dense):
    def __init__(self, scale=1, *args, **kwargs):
        super(DenseSimLoss, self).__init__(*args, **kwargs)
        self.scale = scale  # scale loss

    def get_labels_of_similarity(self, inputs):
        idx = K.arange(0, K.shape(inputs)[0])
        idx_1 = idx[None, :]
        idx_2 = (idx + 1 - idx % 2 * 2)[:, None]
        labels = K.equal(idx_1, idx_2)
        labels = K.cast(labels, K.floatx())
        return labels

    def compute_loss_of_similarity(self, inputs):
        y_true = self.get_labels_of_similarity(inputs)  # 构建标签
        y_pred = K.l2_normalize(inputs, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * self.scale  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def call(self, inputs):
        sim_loss = self.compute_loss_of_similarity(inputs)
        self.add_loss(sim_loss)
        self.add_metric(sim_loss, 'similarity loss')
        return super(DenseSimLoss, self).call(inputs)


# build model
model = build_transformer_model(
    config_path,
    checkpoint_path,
)
output = Lambda(lambda x: x[:, 0])(model.output)
output = DenseSimLoss(scale=1, units=num_classes, activation='softmax')(output)
model = Model(model.inputs, output)
model.summary()


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_val_acc = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == '__main__':
    evaluator = Evaluator('best_clf.weights')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
    model.fit_generator(train_sim_generator.generator(),
                        steps_per_epoch=len(train_sim_generator) * 2,
                        epochs=5,
                        callbacks=[evaluator]
                        )
else:
    model.load_weights('best_clf.weights')
