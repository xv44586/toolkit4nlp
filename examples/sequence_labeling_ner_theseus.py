#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com
import os

import tensorflow as tf
import numpy as np
import re
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import ViterbiDecoder
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.models import Model
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.layers import *
from toolkit4nlp.backend import K, sequence_masking


vocab_dict = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'

data_dir = '/home/mingming.xu/datasets/NLP/ner/company_ner'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'dev.csv')

tokenizer = Tokenizer(vocab_dict,do_lower_case=True)

maxlen = 128
lr = 1e-5
batch_size = 24


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        d = []
        for l in f.split('\n'):
            l = l.strip()
            if not l or len(l) < 1:
                if d:
                    D.append(d)
                d = []
            else:
                if re.search('"', l):
                    segs = l.strip().replace('"', '').split()
                    d.append([' '.join(segs[-1:]), segs[-1]])
                else:
                    d.append(l.split())

        D.append(d)
    return D

train_data = load_data(train_path)
test_data = load_data(test_path)
idx = list(range(len(train_data)))
np.random.shuffle(idx)
train_data = np.array(train_data)

idx_train = [i for i in idx if i % 10 != 0]
idx_valid = [i for i in idx if i % 10 == 0]

valid_data = train_data[idx_valid]
train_data = train_data[idx_train]

# 类别映射
labels = ['O', 'BRA', 'DIR', 'PRO', 'COM', 'P_ADD', 'P_BRA', 'P_COM',
          'ADD', 'ADD1', 'ADD2', 'P_DIR', 'P_ADD1', 'P_PRO', 'P1_BRA', 'P1_DIR', 'P1_COM', 'P_ADD2', 'P1_ADD',
          'ADD3', 'P_ADD3', 'P1_ADD1', 'P1_ADD2', 'P_COM ', 'P1_PRO', 'P2_DIR', 'P2_COM', 'P2_ADD']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 3 + 1


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
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
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


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
x_in = Input(shape=K.int_shape(predecessor.outputs)[1:])
x = Dense(num_labels)(x_in)
CRF = ConditionalRandomField(lr_multiplier=2)
x = CRF(x)
classifier = Model(x_in, x)


from toolkit4nlp.optimizers import extend_with_gradient_accumulation
opt = extend_with_gradient_accumulation(Adam)
opt = opt(learning_rate=5e-6)

predecessor_model = Model(predecessor.inputs, classifier(predecessor.outputs))
predecessor_model.compile(loss=CRF.sparse_loss,
                          optimizer=opt,
                          metrics=[CRF.sparse_accuracy])

predecessor_model.summary()

successor_model = Model(successor.inputs, classifier(successor.outputs))
successor_model.compile(loss=CRF.sparse_loss,
                       optimizer=opt,
                       metrics=[CRF.sparse_accuracy])
successor_model.summary()


def bert_theseus(predecessor, successor, classifier):
    """
    固定predecessor和classifier，随机替换predecessor中的block 为successor来训练successor
    """
    inputs = predecessor.inputs
    # 固定predecessor参数
    for layer in predecessor.model.layers:
        layer.trainable = False
    classifier.trainable = False

    # 随机替换embedding层
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)
    outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])
    # 替换transformer层
    layer_per_block = predecessor.num_hidden_layers // successor.num_hidden_layers
    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layer_per_block):
            predecessor_outputs = predecessor.apply_transformer(predecessor_outputs,
                                                                sub_index + index * layer_per_block)

        successor_outputs = successor.apply_transformer(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])

    outputs = classifier(outputs)
    return Model(inputs, outputs)


theseus_model = bert_theseus(predecessor, successor, classifier)
theseus_model.compile(loss=CRF.sparse_loss,
                      optimizer=opt,
                      metrics=[CRF.sparse_accuracy])
theseus_model.summary()

from toolkit4nlp.utils import ViterbiDecoder


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
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
        for i, label in enumerate(labels):
            if label > 0:
                if label % 3 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 3]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

from tqdm import tqdm


def evaluate(data, model):
    """评测函数
    """
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
    def __init__(self, model_name):
        self.best_val_f1 = 0
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data, self.model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_name)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data, self.model)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__  == '__main__':
    epochs = 1
    batch_size = 16
    train_generator = data_generator(train_data, batch_size)

    predecessor_model_name = 'predecessor_best.weights'
    predecessor_evaluator = Evaluator(predecessor_model_name)

    predecessor_model.fit_generator(
        train_generator.generator(),
        #         steps_per_epoch=len(train_generator),
        steps_per_epoch=20,
        epochs=epochs,
        callbacks=[predecessor_evaluator]
    )
    theseus_model_name = 'theseus_best.weights'
    theseus_evaluator = Evaluator(theseus_model_name)
    theseus_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs * 2,
        callbacks=[theseus_evaluator]
    )
    theseus_model.load_weights(theseus_model_name)

    successor_model_name = 'successor_best.weights'
    successor_evaluator = Evaluator(successor_model_name)
    successor_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[successor_evaluator]
    )