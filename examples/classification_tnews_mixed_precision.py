# -*- coding: utf-8 -*-
# @Date    : 2020/10/20
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_tnews_baseline.py
"""
一行代码开启混合精度，加速训练
best acc: 0.576
tips: 1.只支持tf.train.Optimizer or tf.keras.optimizers.Optimizer继承来的，不支持keras 原生的optimizer
    2. 修改opt 放在build model 前，否则某些情况会报错

训练速度能提高约30% 左右，244ms/step -> 168ms/step
"""
import os
os.environ['TF_KERAS'] = '1'  # 使用tf.keras

import json
from tqdm import tqdm
import numpy as np

from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *

num_classes = 16
maxlen = 128
batch_size = 64
epochs = 5
num_hidden_layers = 12
lr = 1e-5

# BERT base
config_path = '/data/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], l['label_desc']
            label = int(label) - 100 if int(label) < 105 else int(label) - 101
            D.append((text, int(label), label_des))
    return D


# 加载数据集
train_data = load_data(
    '/data/datasets/NLP/CLUE/tnews_public/train.json'
)
valid_data = load_data(
    '/data/datasets/NLP/CLUE/tnews_public/dev.json'
)

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_desc) in self.get_sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([int(label)])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(data=train_data, batch_size=batch_size)
val_generator = data_generator(valid_data, batch_size)

# create opt before build model
opt = Adam(lr)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)  # 开启混合精度

# build model
bert = build_transformer_model(config_path=config_path,
                               checkpoint_path=checkpoint_path,
                               num_hidden_layers=num_hidden_layers)
output = Lambda(lambda x: x[:, 0])(bert.output)
output = Dense(num_classes, activation='softmax')(output)

model = Model(bert.inputs, output)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()

    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(val_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights('best_baseline.weights')
        print('acc: {}, best acc: {}'.format(acc, self.best_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])
else:
    model.load_weights('best_baseline.weights')
