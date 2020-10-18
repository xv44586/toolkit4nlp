# -*- coding: utf-8 -*-
# @Date    : 2020/10/16
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_focal_loss.py
"""
CE best test f1: 77.8
CE_W best test f1: 84.6
FL best test f1: 86.7
数据来源：https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import itertools

import tensorflow as tf
from toolkit4nlp.backend import keras
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import pad_sequences, DataGenerator
from keras.layers import Lambda, Dense
from toolkit4nlp.backend import K
from sklearn.metrics import f1_score


maxlen = 128
batch_size = 32
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('/home/mingming.xu/datasets/NLP/sentiment/sentiment.train.data')
valid_data = load_data('/home/mingming.xu/datasets/NLP/sentiment/sentiment.valid.data')
test_data = load_data('/home/mingming.xu/datasets/NLP/sentiment/sentiment.test.data')

# 构造正负不均衡样本，大致8:1
train_p = [t for t in train_data if t[1] == 1]
train_n = [t for t in train_data if t[1] == 0]
train_new = train_p + train_n[:1000]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.get_sample(shuffle):
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
train_generator = data_generator(train_new, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# 构造模型
def create_model():
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=True,
        num_hidden_layers=1
    )

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.output)
    output = Dense(
        units=1,
        activation='sigmoid',
    )(output)

    model = keras.models.Model(bert.inputs, output)
    return model


def ce_weighted(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=0.15)


def evaluate(data):
    total, right = 0., 0.
    preds, trues = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        preds.append(y_pred)
        trues.append(y_true)

        total += len(y_true)
        right += (y_true == np.round(y_pred)).sum()

    acc = right / total

    preds = np.concatenate(preds)[:, 0]
    trues = np.concatenate(trues)[:, 0]
    f1 = f1_score(y_pred=np.round(preds), y_true=trues, average='weighted')
    return acc, f1


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, save_name='best_model.weights'):
        self.best_val_f1 = 0.
        self.last_test_f1 = 0.
        self.save_name = save_name

    def on_epoch_end(self, epoch, logs=None):
        val_acc, val_f1 = evaluate(valid_generator)
        test_acc, test_f1 = evaluate(test_generator)

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights(self.save_name)
            self.last_test_f1 = test_f1

        print('epoch : %d val acc is: %.5f, test acc: %.5f\n' %(epoch, val_acc, test_acc))
        print(
            u'val_f1: %.5f, best_val_f1: %.5f, test_f1: %.5f\n' %
            (val_f1, self.best_val_f1, test_f1)
        )


def cal_bins(model):
    bins = defaultdict(int)
    outlier = []
    b = np.array([])
    preds = []
    ytrue = []

    for i, d in tqdm(enumerate(train_generator)):
        x, y_true = d
        ytrue.extend(y_true)
        y_pred = model.predict(x)
        m = np.abs(y_pred - y_true)
        preds.append(y_pred)
        if not b.any():
            b = m
        else:
            b = np.concatenate((b, m), axis=0)

        out = np.argwhere(m >= 0.9)
        if out.any():
            outlier.append(i * batch_size + out)
        for i in range(0, 11, 1):
            bins[i / 10] += (np.floor(m * 10) == i).sum()

    yb = np.concatenate(b)
    df = pd.DataFrame({'preds': yb, 'ytrue': ytrue})

    df['preds_int'] = np.floor(df['preds'] * 10)
    df['preds_rank'] = df['preds'].rank(method='dense').astype(int)

    df['pt'] = np.abs(df['preds'] - df['ytrue'])

    p = df[df['ytrue'] == 1]
    n = df[df['ytrue'] == 0]
    pbins = {}
    nbins = {}
    for i in range(11):
        pbins[i] = len(p[p['preds_int'] == i])
        nbins[i] = len(n[n['preds_int'] == i])

    return bins, pbins, nbins


def draw(bins, title=''):
    x = [i / 10 for i in range(0, 11, 1)]
    v = sorted(bins.items())
    v = [i[1] for i in v]
    fig, ax = plt.subplots()
    b = ax.bar(x, v, width=0.1, label='{}'.format(v))

    for a, b in zip(x, v):
        ax.text(a, b + 1, b, ha='left', va='bottom')

    plt.xlim(0, 1)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # train balanced positive and negative model
    model = create_model()
    model.compile(
        loss=ce_weighted,
        optimizer=Adam(1e-4),  # 用足够小的学习率
        metrics=['accuracy'],
    )
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    bins, pbins, nbins = cal_bins(model)
    draw(bins, 'easy vs hard')
    draw(pbins, 'positive easy vs hard')
    draw(nbins, 'negative easy vs hard')


    # train model use focal loss
    def compile_model(alpha, gamma):
        model = create_model()
        model.compile(
            loss=K.focal_loss(alpha, gamma),
            optimizer=Adam(1e-4),  # 用足够小的学习率
            metrics=['accuracy'],
        )
        return model


    # grid search
    result = {}

    alpha = [0.15, 0.25]
    gamma = [2, 1.5]

    for para in itertools.product(alpha, gamma):
        evaluator = Evaluator('best_{}.weights'.format(para))

        al, ga = para
        print('alpha: ', al, ' gamma: ', ga)
        model = compile_model(al, ga)
        model.fit_generator(
            train_generator.generator(),
            steps_per_epoch=len(train_generator),
            epochs=5,
            callbacks=[evaluator]
        )
        result[para] = [evaluator.best_val_f1, evaluator.last_test_f1]

        bins, pbins, nbins = cal_bins(model)
        draw(bins, 'all samples'), draw(pbins, 'positive'), draw(nbins, 'negative')
    print(result)
