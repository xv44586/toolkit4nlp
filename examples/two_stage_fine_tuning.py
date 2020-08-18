# -*- coding: utf-8 -*-
# @Date    : 2020/8/17
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : two_stage_fine_tuning.py
"""
实验验证 bert-of-theseus 中 theseus的必要性：即不添加theseus，训练完predecessor后直接使用其前三层与classifier做fine tuning
结果：即使对classifier 设置不同的学习率，结果依然无法突破bert直接3层fine tuning的结果，所以theseus还是需要的
blog: https://xv44586.github.io/2020/08/09/bert-of-theseus/
"""
import numpy as np
import json

from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

num_classes = 119
maxlen = 128
batch_size = 32


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/train.json'
)
valid_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.get_sample():
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in data:
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
        print('last weights', self.model.layers[-1].layers[-1].weights[1][-10:])
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
        # change random prob
        # has_random = False
        # for l in self.model.layers:
        #     has_random = True
        #     if 'random'in l.name:
        #         print('name ', l.name, 'p: ', l.p)


class ScaleDense(Dense):
    """
    可以设置scale学习率的dense层
    """

    def __init__(self, lr_multiplier=1, **kwargs):
        super(ScaleDense, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier  # 学习率scale参数

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self._kernel = self.add_weight(shape=(input_dim, self.units),
                                       initializer=self.kernel_initializer,
                                       name='kernel',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        if self.use_bias:
            self._bias = self.add_weight(shape=(self.units,),
                                         initializer=self.bias_initializer,
                                         name='bias',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

        if self.lr_multiplier != 1:
            K.set_value(self._kernel, K.eval(self._kernel) / self.lr_multiplier)
            K.set_value(self._bias, K.eval(self._bias) / self.lr_multiplier)

    @property
    def bias(self):
        if self.lr_multiplier != 1:
            return self._bias * self.lr_multiplier
        return self._bias

    @property
    def kernel(self):
        if self.lr_multiplier != 1:
            return self._kernel * self.lr_multiplier
        return self._kernel

    def call(self, inputs):
        return super(ScaleDense, self).call(inputs)


# 加载预训练模型（12层）
predecessor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    prefix='Predecessor-'
)


# 判别模型
x_in = Input(shape=K.int_shape(predecessor.output)[1:])
x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_classes, activation='softmax')(x)
classifier = Model(x_in, x)

predecessor_model = Model(predecessor.inputs, classifier(predecessor.output))
predecessor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_model.summary()


# predecessor_model_3
output = predecessor_model.layers[31].output  # 第3层transform
output = Lambda(lambda x: x[:, 0])(output)
dense = ScaleDense(lr_multiplier=5,
                   units=num_classes,
                   activation='softmax',
                   weights=predecessor_model.layers[-1].get_weights())
output = dense(output)

predecessor_3_model = Model(predecessor_model.inputs, output)
predecessor_3_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_3_model.summary()

if __name__ == '__main__':
    epochs = 5
    # 训练predecessor
    predecessor_evaluator = Evaluator('best_predecessor.weights')
    predecessor_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[predecessor_evaluator]
    )

    # 训练predecessor_3_model
    predecessor_model.load_weights('best_predecessor.weights')
    predecessor_3_evaluator = Evaluator('best_predecessor_3.weights')
    predecessor_3_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[predecessor_3_evaluator]
    )
