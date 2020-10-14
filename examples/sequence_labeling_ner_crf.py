# -*- coding: utf-8 -*-
# @Date    : 2020/8/5
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : sequence_labeling_ner_crf.py
"""
CRF做中文命名实体识别
crf 介绍： https://xv44586.github.io/2019/12/26/from-softmax-to-crf/
数据源：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
标签：BIEO
valid data F1: 95.7%
test data F1: 94.8%
"""

import os
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.utils import ViterbiDecoder, pad_sequences, DataGenerator
from toolkit4nlp.backend import keras, K
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import Adam, extend_with_gradient_accumulation


data_dir = '/home/mingming.xu/datasets/NLP/ner/china-people-daily-ner-corpus/'
train_path = os.path.join(data_dir, 'example.train')
test_path = os.path.join(data_dir, 'example.test')
val_path = os.path.join(data_dir, 'example.dev')

vocab_dict = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
tokenizer = Tokenizer(vocab_dict, do_lower_case=True)

# 类别映射
labels = ['PER', 'LOC', 'ORG']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 3 + 1

bert_layers = 12  # 1-12 ，越小越靠前，相应的learning rate 也要相应调大
maxlen = 256
epochs = 10
batch_size = 16
crf_lr_multiplier = 2  # 放大crf 层loss
learning_rate = 1e-5
grad_accum_steps = 2  # 模拟batch_size=32


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


train_data = load_data(train_path)
test_data = load_data(test_path)
valid_data = load_data(val_path)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample(shuffle):
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


model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

opt = extend_with_gradient_accumulation(Adam)
para = {'learning_rate': learning_rate, 'grad_accum_steps': grad_accum_steps}
opt = opt(**para)

model.compile(
    loss=CRF.sparse_loss,
    optimizer=opt,
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
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


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('./best_model.weights')
