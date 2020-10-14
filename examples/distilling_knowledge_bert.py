# -*- coding: utf-8 -*-
# @Date    : 2020/9/8
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : distilling_knowledge_bert.py
"""
通过利用大模型（或ensemble多个模型）得到一个 Teacher Model，利用 Teacher Model 来对数据生成 soften labels，
然后通过一个小模型，同时学习 Ground True 和 soften label，来做知识蒸馏。
在生成 soften labels 时，有一个超参数 Temperature，该参数用来平滑 soften labels 的分布，可以看做是一种正则化约束。

实验：中文文本分类
数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
bert-12: 60.21%
student-3: 58.14%
teacher-student:60.14%
student-3-self-kd: 59.6%
teacher-12-self-kd:61.07%
normal-noise-bert-3:58.4%

blog: [distilling knowledge of bert](https://xv44586.github.io/2020/08/31/bert-01/)

ref:
  - [Distilling the Knowledge in a Neural Network](http://arxiv.org/abs/1503.02531)
  - [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](http://arxiv.org/abs/1903.12136)
  - [TinyBERT: Distilling BERT for Natural Language Understanding](http://arxiv.org/abs/1909.10351)
  - [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](http://arxiv.org/abs/1910.01108)
"""

import json
from tqdm import tqdm

import numpy as np
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.optimizers import Adam, extend_with_piecewise_linear_lr
from toolkit4nlp.utils import DataGenerator, pad_sequences
from toolkit4nlp.layers import *
from keras.models import Model
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.losses import kullback_leibler_divergence

num_classes = 119
maxlen = 128
batch_size = 32
Temperature = 3  # 平滑soften labels 分布，越大越平滑，一般取值[1, 10]

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


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


class DataGenerator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.get_sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if type(label) != list and type(label) != np.array:
                label = [label]
            elif type(label) == np.array:
                label = list(label)
            batch_labels.append(label)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
y_train = [d[1] for d in train_data]
y_train = to_categorical(y_train)
train_data = [[d[0], y_train[i].tolist()] for i, d in enumerate(train_data)]
train_generator = DataGenerator(data=train_data, batch_size=batch_size)
valid_generator = DataGenerator(data=valid_data, batch_size=batch_size)


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true)[:, :num_classes].argmax(axis=1)
        if y_true.shape[1] > 1:
            y_true = y_true[:, :num_classes].argmax(axis=-1)
        else:
            y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename, valid_model=None):
        self.best_val_acc = 0.
        self.savename = savename
        self.valid_model = valid_model

    def on_epoch_end(self, epoch, logs=None):
        self.valid_model = self.valid_model or self.model
        val_acc = evaluate(valid_generator, self.valid_model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


# teacher model（12层）
teacher = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=True,
    num_hidden_layers=12,
    prefix='Teacher-'
)
output = Lambda(lambda x: x[:, 0])(teacher.output)
logits = Dense(num_classes)(output)
soften = Activation(activation='softmax')(logits)
teacher_logits = Model(teacher.inputs, logits)
teacher_soften = Model(teacher.inputs, soften)
teacher_soften.compile(loss='categorical_crossentropy', optimizer=Adam(2e-5), metrics=['acc'])
teacher_soften.summary()


class StudentDataGenerator(DataGenerator):
    """数据生成器
    """

    def __iter__(self):
        batch_token_ids, batch_segment_ids, batch_labels, batch_logits = [], [], [], []
        for is_end, (text, label, logits) in self.get_sample():
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            batch_labels.append(label)
            batch_logits.append(logits)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                batch_logits = pad_sequences(batch_logits)
                yield [batch_token_ids, batch_segment_ids], [batch_labels, batch_logits]

                batch_token_ids, batch_segment_ids, batch_labels, batch_logits = [], [], [], []


# student model
student = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    num_hidden_layers=3,
    prefix='Student-'
)
output = Lambda(lambda x: x[:, 0])(student.output)
s_logits = Dense(num_classes)(output)
s_soften = Activation(activation='softmax')(s_logits)
student_model = Model(student.inputs, s_soften)

s_logits_t = Lambda(lambda x: x / Temperature)(s_logits)
s_logits_t = Activation(activation='softmax')(s_logits_t)
# soften_logits = concatenate([s_soften, s_logits_t])
# student_train = Model(student.inputs, soften_logits)

student_train = Model(student.inputs, [s_soften, s_logits_t])

student_train.summary()


def normal_noise(label, scale=0.1):
    # add normal noise to create a fake soften labels
    normal_noise = np.random.normal(scale=scale, size=(num_classes,))
    new_label = label + normal_noise
    new_label = K.softmax(new_label / Temperature).numpy()
    return new_label


if __name__ == '__main__':
    # train teacher model
    teacher_evaluator = Evaluator('best_teacher.weights')
    teacher_soften.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[teacher_evaluator]
    )

    # create soften labels
    teacher_soften.load_weights('best_teacher.weights')

    y_train_logits = []
    y_train = []
    for x, label in tqdm(train_generator):
        y_train_logits.append(teacher_logits.predict(x))
        y_train.append(label)

    y_train_logits = np.concatenate(y_train_logits)
    y_train = np.concatenate(y_train)
    y_soften = K.softmax(y_train_logits / Temperature).numpy()
    new_y_train = np.concatenate([y_train, y_soften], axis=-1)

    # create normal noise fake soften labels datasets
    # new_data = [[d[0], d[1], normal_noise(d[1])] for d in train_data]
    # student_data_generator = StudentDataGenerator(new_data, batch_size)

    # create new datasets
    new_data = [[d[0], d[1], y_train_logits[i].tolist()] for i, d in enumerate(train_data)]
    student_data_generator = StudentDataGenerator(new_data, batch_size)

    # check soften labels accuracy
    if_correct = [np.array(d[1]).argmax() == np.array(d[2]).argmax() for d in new_data]
    correct = [t for t in if_correct if t]
    print('soften labels acc is: ', float(len(correct)) / len(if_correct))

    # train student model
    student_evaluator = Evaluator('best_student_train.weights', student_model)
    student_train.compile(loss=['categorical_crossentropy', kullback_leibler_divergence],
                          optimizer=Adam(2e-5),
                          metrics=['acc'],
                          loss_weights=[1, Temperature ** 2]  # 放大 kld
                          )
    student_train.fit_generator(
        student_data_generator.generator(),
        steps_per_epoch=len(student_data_generator),
        epochs=10,
        callbacks=[student_evaluator]
    )
else:
    student_model.load_weights('best_student_train.weights')
