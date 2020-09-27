# -*- coding: utf-8 -*-
# @Date    : 2020/8/11
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : classification_ifytek_bert_of_theseus.py
# ! -*- coding:utf-8 -*-
"""
bert-of-theseus 方法在文本分类例子下的模型压缩
数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
bert-12: 0.607
bert-3: 0.579
bert-6: 0.591
theseus-3: 0.595
ref:
  - https://arxiv.org/abs/2002.02925
  - https://kexue.fm/archives/7575
  - https://zhuanlan.zhihu.com/p/112787764
"""
"""
新增 “加法” 模型，即不再使用随机替换的方式训练theseus_model,而采用将对应输出按比例相加的方式来训练。
原因：复现时，随机的方式结果不稳定，有时得到的效果可能与直接fine-tuning类似，用加法模型结果更稳定一些
"""

import json
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import Adam
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


class BinaryRandomChoice(Layer):
    """随机二选一
    """

    def __init__(self, **kwargs):
        super(BinaryRandomChoice, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    def call(self, inputs):
        source, target = inputs
        mask = K.random_binomial(shape=[1], p=0.5)
        output = mask * source + (1 - mask) * target
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class ProportionalAdd(Layer):
    """将两层的结果乘比例后相加，output = (input_1 * proportion + input_2 * (1 - proportion)) / 2
    """

    def __init__(self, proportion=0.5, **kwargs):
        super(ProportionalAdd, self).__init__(**kwargs)
        self.supports_masking = True
        self.proportion = proportion

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    def call(self, inputs):
        source, target = inputs
        source = source * self.proportion
        target = target * (1 - self.proportion)
        output = (source + target)/2
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def bert_of_theseus(predecessor, successor, classifier):
    """bert of theseus
    """
    inputs = predecessor.inputs
    # 固定住已经训练好的predecessor 和 classifier
    for layer in predecessor.model.layers:
        layer.trainable = False
    classifier.trainable = False
    # Embedding层替换
    # 也可以固定住Embedding
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)
    outputs = ProportionalAdd()([predecessor_outputs, successor_outputs])
    # Transformer层替换，用successor 的 layer 替换对应predecessor的module
    layers_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers
    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layers_per_module):
            predecessor_outputs = predecessor.apply_transformer_layers(
                predecessor_outputs, layers_per_module * index + sub_index
            )
        successor_outputs = successor.apply_transformer_layers(outputs, index)
        outputs = ProportionalAdd()([predecessor_outputs, successor_outputs])
    # 返回模型
    outputs = classifier(outputs)
    model = Model(inputs, outputs)
    return model


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


# 加载预训练模型（12层）
predecessor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    prefix='Predecessor-'
)

# 加载预训练模型（3层）
successor = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=3,
    prefix='Successor-'
)

# 判别模型
x_in = Input(shape=K.int_shape(predecessor.output)[1:])
x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_classes, activation='softmax')(x)
classifier = Model(x_in, x)

predecessor_model = Model(predecessor.inputs, classifier(predecessor.output))
predecessor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_model.summary()

successor_model = Model(successor.inputs, classifier(successor.output))
successor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
successor_model.summary()

theseus_model = bert_of_theseus(predecessor, successor, classifier)
theseus_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
theseus_model.summary()


def print_trainable(model):
    print(model.name)
    print([l.trainable for l in model.layers[-1].layers])


if __name__ == '__main__':
    # 训练predecessor
    predecessor_evaluator = Evaluator('best_predecessor.weights')
    predecessor_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[predecessor_evaluator]
    )

    # 训练theseus
    theseus_evaluator = Evaluator('best_theseus.weights')
    theseus_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[theseus_evaluator]
    )
    theseus_model.load_weights('best_theseus.weights')

    # 训练successor
    successor_evaluator = Evaluator('best_successor.weights')
    successor_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[successor_evaluator]
    )
