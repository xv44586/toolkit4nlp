# -*- coding: utf-8 -*-
# @Date    : 2020/7/29
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : qa_dgcnn_example.py
"""
bert + dgcnn fine-tuning to do qa task
数据是百度2020icl比赛的机器阅读(http://lic2020.cipsc.org.cn/)
F1: 82左右
"""
import json, os

import numpy as np
import tensorflow as tf
from toolkit4nlp.backend import keras, K
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.optimizers import Adam, extend_with_gradient_accumulation, extend_with_weight_decay
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import Layer, Dense, Permute, Input, Layer, Lambda, Dropout
from toolkit4nlp.layers import AttentionPooling1D, DGCNN, SinCosPositionEmbedding
from toolkit4nlp.models import Model
from tqdm import tqdm

K.clear_session()
# 基本信息
maxlen = 512
epochs = 5
batch_size = 4
learning_rate = 2e-5

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    return D


# 读取数据
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/train.json'
)
val_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def split_context(context, max_context_len):
    for i in range(0, len(context) - max_context_len + 1):
        yield context[i: i + max_context_len]


class data_generator(DataGenerator):
    """数据生成器
    """

    def random_generator(self, s):
        l = maxlen // 2 + maxlen % 2

        if len(s) > l:
            p = np.random.random()
            if p > 0.5:
                #                 i = np.random.randint(len(s) - l +1)
                #                 j = np.random.randint(l + i, min(len(s), maxlen) + 1)
                i = np.random.randint(len(s) - l + 1)
                j = np.random.randint(l + i, min(len(s), i + maxlen) + 1)

                return s[i:j]
            else:

                return s[: maxlen]
        else:
            return s

    def random_padding(self, tokens, start_idx):
        pad_rate = 0.05
        p = np.random.random()
        if p < 0.5:
            return tokens

        c = len(tokens)
        pad_c = int((c - 1 - start_idx) * pad_rate)
        pad_idx = np.random.choice(range(c - 1), pad_c)

        for idx in set(pad_idx):
            tokens[idx] = tokenizer._token_pad_id

        return tokens

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample(shuffle):
            context, question, answers = item[1:]

            context = self.random_generator(context)

            token_ids, segment_ids = tokenizer.encode(question, context, maxlen=maxlen)
            qt = tokenizer.tokenize(question)
            token_ids = self.random_padding(token_ids, len(qt))

            a = np.random.choice(answers)
            a_token_ids = tokenizer.encode(a)[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = pad_sequences(batch_token_ids)
                    batch_segment_ids = pad_sequences(batch_segment_ids)
                    batch_labels = pad_sequences(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class ConcatSeq2Vec(Layer):
    def __init__(self, **kwargs):
        super(ConcatSeq2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatSeq2Vec, self).build(input_shape)

    def call(self, x):
        seq, vec = x
        vec = K.expand_dims(vec, 1)
        vec = K.tile(vec, [1, K.shape(seq)[1], 1])
        return K.concatenate([seq, vec], 2)

    def compute_mask(self, inputs, mask):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (input_shape[0][-1] + input_shape[1][-1],)


class MaskedSoftmax(Layer):
    """在序列长度那一维进行softmax，并mask掉padding部分
    """

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)


# build model
model = build_transformer_model(
    config_path,
    checkpoint_path,
)

inputs = [Input(shape=K.int_shape(model.inputs[0])[1:]), Input(shape=K.int_shape(model.inputs[1])[1:])]
output = model(inputs)
output = SinCosPositionEmbedding(K.int_shape(output)[-1])(output)

output = Dropout(0.5)(output)
output = Dense(384, activation='tanh')(output)

att = AttentionPooling1D(name='attention_pooling_1')(output)

output = ConcatSeq2Vec()([output, att])

output = DGCNN(dilation_rate=1, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=2, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=5, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=8, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=16, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=8, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=5, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=2, drop_rate=0.1)(output)
output = DGCNN(dilation_rate=1, drop_rate=0.1)(output)
output = SinCosPositionEmbedding(K.int_shape(output)[-1])(output)
att = AttentionPooling1D()(output)
output = ConcatSeq2Vec()([output, att])
# att = K.expand_dims(att, 1)
# output = Add()([output, att])

output = Dropout(0.3)(output)

output = Dense(2)(output)
output = MaskedSoftmax()(output)
output = Permute((2, 1), name='permute')(output)

model = Model(inputs, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


# optimizer
optimizer = extend_with_weight_decay(Adam)
optimizer = extend_with_gradient_accumulation(optimizer)
params = {
    'learning_rate': learning_rate,
    'weight_decay_rate': 1e-5,
    'exclude_from_weight_decay': ['norm', 'bias'],
    'grad_accum_steps': 4
}
optimizer = optimizer(**params)
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=optimizer,
    metrics=[sparse_accuracy]
)


def extract_answer(question, context, max_a_len=32):
    """抽取答案函数
    """
    max_q_len = 64
    q_token_ids = tokenizer.encode(question, maxlen=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, maxlen=maxlen - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    token_ids = np.array([token_ids])  # tf2.X 必须要转np.array
    segment_ids = np.array([segment_ids])
    probas = model.predict([token_ids, segment_ids])[0]
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]


def predict_to_file(infile, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1])
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()


def evaluate(filename):
    """评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json')
    metrics = json.loads(
        os.popen(
            'python /home/mingming.xu/datasets/NLP/qa/dureader_robust-data/evaluate.py %s %s'
            % (filename, filename + '.pred.json')
        ).read().strip()
    )
    return metrics


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(
            '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/dev.json'
        )
        if float(metrics['F1']) >= float(self.best_val_f1):
            self.best_val_f1 = metrics['F1']
            model.save_weights('best_model.weights')
        metrics['BEST F1'] = self.best_val_f1
        print(metrics)


if __name__ == '__main__':
    train_generator = data_generator(train_data, batch_size)

    evaluator = Evaluator()
    model.fit(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator]
    )
else:
    model.load_weights('best_model.weights')
