# -*- coding: utf-8 -*-
# @Date    : 2020/10/22
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_pet_seq2seq.py
"""
seq2seq用来分类，解码时不使用beam search而采用pet 的方式解码，缩小解码空间，提高准确率

"""
import json
from tqdm import tqdm
import numpy as np

from toolkit4nlp.backend import keras, K, is_tf_keras
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *

num_classes = 16
maxlen = 128
batch_size = 32
max_label = 2

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

desc2label = {
    'news_agriculture': '农业',
    'news_car': '汽车',
    'news_culture': '文化',
    'news_edu': '教育',
    'news_entertainment': '娱乐',
    'news_finance': '财经',
    'news_game': '游戏',
    'news_house': '房产',
    'news_military': '军事',
    'news_sports': '体育',
    'news_stock': '股市',
    'news_story': '民生',
    'news_tech': '科技',
    'news_travel': '旅游',
    'news_world': '国际'
}


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], desc2label[l['label_desc']]
            label = int(label) - 100 if int(label) < 105 else int(label) - 101
            D.append((text, int(label), label_des))
    return D


# 加载数据集
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/tnews_public/train.json'
)
valid_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/tnews_public/dev.json'
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

    def __init__(self, seq2seq=False, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        self.seq2seq = seq2seq

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.get_sample(shuffle):
            if not self.seq2seq:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            else:
                text_token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
                label_token_ids = tokenizer.encode(label_des, maxlen=max_label + 2)[0][1:-1]
                token_ids = text_token_ids + label_token_ids
                segment_ids = [0] * len(text_token_ids) + [1] * len(label_token_ids)
                batch_labels.append([label])

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                if self.seq2seq:
                    yield [batch_token_ids, batch_segment_ids], None
                else:
                    yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


id2label = dict([(d[1], d[2]) for d in train_data])
labels = [i[1] for i in sorted(id2label.items())]

# label prob
char1 = [w[0] for w in labels]
char2 = [w[1] for w in labels]

char1_idx = tokenizer.encode(''.join(char1))[0][1:-1]
char2_idx = tokenizer.encode(''.join(char2))[0][1:-1]

# 转换数据集
train_generator = data_generator(data=train_data[:2000], batch_size=batch_size, seq2seq=True)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)


class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# build model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


def pet_decode(inputs):
    # 减少搜索空间，只在label可能的token上进行搜索
    token_ids, segment_ids = tokenizer.encode(inputs, maxlen=maxlen)
    p1 = model.predict([[token_ids], [segment_ids]])[:, -1, char1_idx]

    char1_token_ids = np.expand_dims(char1_idx, 1)
    segment_ids = segment_ids + [1]

    token_ids = np.tile(token_ids, (len(labels), 1))
    token_ids = np.concatenate([token_ids, char1_token_ids], axis=1)
    segment_ids = np.tile(segment_ids, (len(labels), 1))
    p2 = model.predict([token_ids, segment_ids])[:, -1, char2_idx]
    p1 = np.reshape(p1, (-1, 1))
    label_idx = np.diagonal(p1 + p2).argmax()
    return labels[label_idx]


def just_show():
    idx = np.random.choice(len(train_data), 3)
    for i in idx:
        sample = train_data[i]
        print(u'context：%s' % sample[0])
        print(u'label id：%s ' % sample[1])
        print(u'label desc: %s' % sample[2])
        new_label = pet_decode(sample[0])
        print('pet label: %s ' % new_label)


def evaluate(data=valid_data):
    total, right = 0., 0.
    for x, _, y_true in tqdm(data):
        total += 1
        pred = pet_decode(x)
        if pred == y_true:
            right += 1
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path='best_model.weights'):
        self.lowest_loss = 1e10
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        just_show()
        if logs['loss'] <= self.lowest_loss:
            self.lowest_loss = logs['loss']
            self.model.save_weights(self.save_path)

        print('current loss :{}, lowest loss: {}'.format(logs['loss'], self.lowest_loss))


if __name__ == '__main__':
    # zero-shot
    zero_acc = evaluate(valid_data)
    print('zero shot acc: ', zero_acc)

    # few shot
    evaluator = Evaluator()
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=5,
                        callbacks=[evaluator])
    acc = evaluate()
    print('few shot acc: ', acc)
