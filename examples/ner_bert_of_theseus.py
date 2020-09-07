# -*- coding: utf-8 -*-
# @Date    : 2020/8/10
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : bert_of_theseus.py
# ! -*- coding:utf-8 -*-
# china daily ner data
# 方法为BERT-of-Theseus
# 论文：https://arxiv.org/abs/2002.02925
#

import os

os.environ['TF_KERAS'] = '1'

import json
from tqdm import tqdm
import numpy as np
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.optimizers import Adam, extend_with_piecewise_linear_lr
from toolkit4nlp.utils import pad_sequences, DataGenerator, ViterbiDecoder
from keras.layers import Input, Lambda, Dense, Layer
from keras.models import Model

maxlen = 256
batch_size = 16

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

# data
data_dir = '/home/mingming.xu/datasets/NLP/ner/china-people-daily-ner-corpus/'
train_path = os.path.join(data_dir, 'example.train')
test_path = os.path.join(data_dir, 'example.test')
val_path = os.path.join(data_dir, 'example.dev')


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 加载数据集
train_data = load_data(train_path)
test_data = load_data(test_path)
valid_data = load_data(val_path)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['PER', 'LOC', 'ORG']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 3 + 1


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 3 + 1
                        I = label2id[l] * 3 + 2
                        E = label2id[l] * 3 + 3

                        labels += ([B] + [I] * (len(w_token_ids) - 2) + [E] * int(len(w_token_ids) > 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                # batch_labels = np.expand_dims(batch_labels, -1)
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


def bert_of_theseus(predecessor, successor, classfier):
    """bert of theseus
    """
    inputs = predecessor.inputs
    # 固定住已经训练好的层
    for layer in predecessor.model.layers:
        layer.trainable = False
    classfier.trainable = False
    # Embedding层替换
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)
    outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # Transformer层替换
    layers_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers
    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layers_per_module):
            predecessor_outputs = predecessor.apply_main_layers(
                predecessor_outputs, layers_per_module * index + sub_index
            )
        successor_outputs = successor.apply_main_layers(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # 返回模型
    outputs = classfier(outputs)
    model = Model(inputs, outputs)
    return model


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def decode(self, nodes):
        #         nodes = nodes[1:-1]
        return np.argmax(nodes, axis=-1)

    def recognize(self, text, model):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids = np.array([token_ids])
        segment_ids = np.array([segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels[1:-1]):
            if label > 0:
                if label % 3 == 1:
                    starting = True
                    entities.append([[i + 1], id2label[(label - 1) // 3]])
                elif starting:
                    entities[-1][0].append(i + 1)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


# NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
NER = NamedEntityRecognizer(trans=np.zeros((num_labels, num_labels)), starts=[0], ends=[0])


def evaluate(data, model):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text, model))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_val_f1 = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        print('last weights', self.model.layers[-1].layers[-1].weights[1][-10:])
        f1, precision, recall = evaluate(valid_data, self.model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.savename)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data, self.model)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
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
# x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_labels, activation='softmax')(x_in)
classfier = Model(x_in, x)

predecessor_model = Model(predecessor.inputs, classfier(predecessor.output))
predecessor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
predecessor_model.summary()

successor_model = Model(successor.inputs, classfier(successor.output))
successor_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
successor_model.summary()

theseus_model = bert_of_theseus(predecessor, successor, classfier)
theseus_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
theseus_model.summary()


def print_trainable(model):
    print(model.name)
    print([l.trainable for l in model.layers[-1].layers])


print_trainable(predecessor_model)
print_trainable(successor_model)
print_trainable(theseus_model)

if __name__ == '__main__':
    # 训练predecessor
    predecessor_evaluator = Evaluator('best_predecessor.weights')
    predecessor_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[predecessor_evaluator]
    )

    print(theseus_model.layers[-1].trainable)
    # for layer in theseus_model.layers[-1].layers:
    #   layer.trainable = False

    # 训练theseus
    theseus_evaluator = Evaluator('best_theseus.weights')
    theseus_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[theseus_evaluator]
    )
    theseus_model.load_weights('best_theseus.weights')

    print_trainable(successor_model)
    # for layer in successor_model.layers[-1].layers:
    #   layer.trainable = True
    print_trainable(successor_model)
    # 训练successor
    successor_evaluator = Evaluator('best_successor.weights')
    successor_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[successor_evaluator]
    )
