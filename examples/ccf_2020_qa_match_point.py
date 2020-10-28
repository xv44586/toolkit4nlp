# -*- coding: utf-8 -*-
# @Date    : 2020/10/28
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : ccf_2020_qa_match_point.py
"""
主要思路：将reply顺序拼接到query后面，利用每个reply的[SEP]token做二分类
best val f1: 78.7
"""
import os
from tqdm import tqdm
import numpy as np

from toolkit4nlp.utils import *
from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.backend import *

batch_size = 16
maxlen = 280
epochs = 10
lr = 1e-5

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm//vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

path = '/home/mingming.xu/datasets/NLP/ccf_qa_match/'


def load_data(train_test='train'):
    D = {}
    with open(os.path.join(path, train_test, train_test + '.query.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            D[span[0]] = {'query': span[1], 'reply': []}

    with open(os.path.join(path, train_test, train_test + '.reply.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            if len(span) == 4:
                q_id, r_id, r, label = span
            else:
                label = None
                q_id, r_id, r = span
            D[q_id]['reply'].append([r_id, r, label])
    d = []
    for k, v in D.items():
        v.update({'query_id': k})
        d.append(v)
    return d


train_data = load_data('train')
test_data = load_data('test')


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_mask, batch_labels = [], [], [], []
        for is_end, item in self.get_sample(shuffle):
            query = item['query']
            reply = item['reply']
            token_ids, segment_ids = tokenizer.encode(query)
            mask_ids, label_ids = segment_ids[:], segment_ids[:]
            for rp in reply:
                _, r, label = rp
                r_token_ids = tokenizer.encode(r)[0][1:]
                r_segment_ids = [1] * len(r_token_ids)
                r_mask_ids = [0] * (len(r_token_ids) - 1) + [1]  # 每句的句尾sep作为特征
                r_label_ids = r_mask_ids[:]
                if int(label) == 0:
                    r_label_ids[-1] = 0

                token_ids += r_token_ids
                segment_ids += r_segment_ids
                mask_ids += r_mask_ids
                label_ids += r_label_ids

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask.append(mask_ids)
            batch_labels.append(label_ids)

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_mask = pad_sequences(batch_mask)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels, batch_mask], None

                batch_token_ids, batch_segment_ids, batch_mask, batch_labels = [], [], [], []


train_generator = data_generator(train_data[:5000], batch_size)
valid_generator = data_generator(train_data[5000:], batch_size)
test_generator = data_generator(test_data, batch_size)


class PointLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        y_pred, y_true, label_mask = inputs
        loss = K.binary_crossentropy(y_true, y_pred)
        loss = K.sum(loss * label_mask) / K.sum(label_mask)
        return loss


class ClsMerge(Layer):
    def call(self, inputs):
        input_shape = K.shape(inputs)
        cls = inputs[:, 0]
        cls = K.expand_dims(cls, 1)
        cls = K.tile(cls, [1, input_shape[1], 1])

        return K.concatenate([inputs, cls], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)

    def compute_mask(self, inputs, mask=None):
        return mask


bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha'
)

m_inputs = Input(shape=(None,))
label_inputs = Input(shape=(None,))
output = bert.output
output = ClsMerge()(output)
output = Dropout(0.5)(output)
output = Dense(768, activation='tanh')(output)

output = Dense(1, activation='sigmoid')(output)

output = Lambda(lambda x: x[:, :, 0])(output)

x = PointLoss(0)([output, label_inputs, m_inputs])

train_model = Model(bert.inputs + [label_inputs, m_inputs], x)

infer_model = Model(bert.inputs, output)
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()


def extract_label(labels, label_masks):
    """
    从label序列中提取出每个回复对应的label
    """
    labels = labels[label_masks > 0]
    labels = list(labels)

    p = []
    s, e = 0, 0
    for lm in label_masks:
        e += lm.sum()
        p.append(labels[s:e])
        s = e
    return p


def predict(item):
    '''
    获取对应回复的label
    '''
    token_ids, segment_ids, label_mask = item
    pred = infer_model.predict([token_ids, segment_ids])
    pred = np.round(pred)

    return extract_label(pred, label_mask)


def evaluate(data=valid_generator):
    P, R, TP = 0., 0., 0.
    for (tokens, segments, labels, label_masks), _ in tqdm(data):
        y_pred = predict([tokens, segments, label_masks])
        y_true = extract_label(labels, label_masks)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        R += y_pred.sum()
        P += y_true.sum()
        TP += ((y_pred + y_true) > 1).sum()

    print(P, R, TP)
    pre = TP / R
    rec = TP / P

    return 2 * (pre * rec) / (pre + rec)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_f1 = 0.

    def on_epoch_end(self, eopch, logs=None):
        f1 = evaluate(valid_generator)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights('best_model.weights')

        print('f1: {}, best f1: {}'.format(f1, self.best_f1))


if __name__ == '__main__':
    evaluator = Evaluator()
    train_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
else:
    infer_model.load_weights('best_model.weights')
