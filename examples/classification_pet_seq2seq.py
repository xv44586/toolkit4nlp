# -*- coding: utf-8 -*-
# @Date    : 2020/10/22
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_pet_seq2seq.py
"""
seq2seq用来分类，解码时不直接全字符匹配，而采用pet 相同方式，缩小解码空间

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
labels = list(desc2label.values())


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
            #                 batch_labels.append(label_token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                #                 if self.seq2seq:
                #                     yield [batch_token_ids, batch_segment_ids], None
                #                 else:
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


id2label = dict([(d[1], d[2]) for d in train_data])
# label prob
char1 = [w[0] for w in labels]
char2 = [w[1] for w in labels]

char1_idx = tokenizer.encode(''.join(char1))[0][1:-1]
char2_idx = tokenizer.encode(''.join(char2))[0][1:-1]

# 转换数据集
train_generator = data_generator(data=train_data[:2000], batch_size=batch_size, seq2seq=True)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)


class TotalLoss(Loss):
    "计算两部分loss：seq2seq的交叉熵，probs与true label的交叉熵"

    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_loss_of_seq2seq(inputs, mask)
        #         classification_loss = self.compute_loss_of_classification(inputs, mask)
        self.add_metric(seq2seq_loss, name='seq2seq_loss')
        #         self.add_metric(classification_loss, name='classification_loss')
        #         acc = self.compute_classification_acc(inputs, mask)
        #         self.add_metric(acc, name='acc')
        return seq2seq_loss

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token
        y_pred = y_pred[:, :-1]  # 错开一位
        y_mask = y_mask[:, 1:]  # 利用segment_ids mask掉第一个segment
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss * 0.2

    def compute_loss_of_classification(self, inputs, mask=None):
        _, _, y_pred, _, y_true = inputs
        return K.sparse_categorical_crossentropy(y_true, y_pred)

    def compute_classification_acc(self, inputs, mask=None):
        _, _, y_pred, _, y_true = inputs
        equal = K.equal(K.cast(K.argmax(y_pred, axis=-1), 'int32'), K.cast(y_true, 'int32'))
        return K.cast(equal, K.floatx()) / K.cast(K.shape(y_true)[0], K.floatx())


bert = build_transformer_model(checkpoint_path=checkpoint_path,
                               config_path=config_path,
                               with_pool='linear',
                               application='unilm',
                               keep_tokens=keep_tokens,
                               return_keras_model=False)
label_inputs = Input(shape=(None,), name='label_inputs')

pooler = bert.model.outputs[0]
classification_output = Dense(units=num_classes, activation='softmax', name='classifier')(pooler)
classifier = Model(bert.model.inputs, classification_output)

seq2seq = Model(bert.model.inputs, bert.model.outputs[1])

outputs = TotalLoss([2])(bert.model.inputs + bert.model.outputs)
# outputs = Dense(num_classes, activation='softmax')(outputs)
train_model = Model(bert.model.inputs, [classification_output, outputs])

train_model.summary()
train_model.compile(loss=['sparse_categorical_crossentropy', None], optimizer=Adam(1e-5), metrics=['acc'])


def evaluate(val_data=valid_generator):
    total = 0.
    right = 0.
    for x, y_true in tqdm(val_data):
        y_pred = classifier.predict(x).argmax(axis=-1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    print(total, right)
    return right / total


def pet_decode(inputs):
    token_ids, segment_ids = tokenizer.encode(inputs, maxlen=maxlen)
    p1 = seq2seq.predict([[token_ids], [segment_ids]])[:, -1, char1_idx]

    char1_token_ids = np.expand_dims(char1_idx, 1)
    segment_ids = segment_ids + [1]

    token_ids = np.tile(token_ids, (len(labels), 1))
    token_ids = np.concatenate([token_ids, char1_token_ids], axis=1)
    segment_ids = np.tile(segment_ids, (len(labels), 1))
    p2 = seq2seq.predict([token_ids, segment_ids])[:, -1, char2_idx]
    label = np.diagonal(p1 * p2).argmax()
    return label


def evaluate_pet(data=valid_data):
    total, right = 0., 0.
    for x, y_true, desc in tqdm(data):
        total += 1
        pred = pet_decode(x)
        if pred == int(y_true):
            right += 1
    return total / right


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path='best_model.weights'):
        self.best_val_acc = 0.
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate_pet(valid_data)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.save_path)

        print('current acc :{}, best val acc: {}'.format(val_acc, self.best_val_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    train_model.fit_generator(train_generator.generator(),
                              steps_per_epoch=len(train_generator),
                              epochs=5,
                              callbacks=[evaluator])
    # 切换解码方式评估
    evaluate(train_data)
