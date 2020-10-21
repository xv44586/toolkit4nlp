# -*- coding: utf-8 -*-
# @Date    : 2020/10/19
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_adabelief.py
"""
classification use AdaBelief:

bert-3-Adam: 57.9
bert-3-AdaBelief: 58.4

ref: [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/pdf/2010.07468.pdf)
"""
import json
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import AdaBelief
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import Input, Lambda, Dense, Layer

num_classes = 119
maxlen = 128
batch_size = 32

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
        val_acc = evaluate(valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.savename)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


# 加载预训练模型（3层）
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=3,
    prefix='Successor-'
)
x = Lambda(lambda x: x[:, 0])(bert.output)
x = Dense(units=num_classes, activation='softmax')(x)
model = Model(bert.inputs, x)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=AdaBelief(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
model.summary()

if __name__ == '__main__':
    # 训练
    evaluator = Evaluator('best_model.weights')
    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator]
    )

else:
    model.load_weights('best_model.weights')