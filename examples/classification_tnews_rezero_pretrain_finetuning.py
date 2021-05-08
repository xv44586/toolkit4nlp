# -*- coding: utf-8 -*-
# @Date    : 2021/4/14
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : classification_tnews_rezero_pretrain_fintuning.py
"""

环境：tf1.X + tf.keras
"""
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import json
from tqdm import tqdm
import numpy as np
import jieba

from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import *
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *


VERSION = 'rezero'  # rezero/pre/post
num_classes = 16
maxlen = 128
batch_size = 32
pretrain_epochs = 50
fine_tune_epochs = 5
pretrain_lr = 5e-5
fine_tune_lr = 1e-5

# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], l['label_desc']
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

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

all_data = train_data + valid_data
pretrain_data = [d[0] for d in all_data]

# whole word mask
pretrain_data = [jieba.lcut(d) for d in pretrain_data]


def random_masking(lines):
    """对输入进行随机mask, 支持多行
    """
    if type(lines[0]) != list:
        lines = [lines]

    sources, targets = [tokenizer._token_start_id], [0]
    segments = [0]

    for i, sent in enumerate(lines):
        source, target = [], []
        segment = []
        rands = np.random.random(len(sent))
        for r, word in zip(rands, sent):
            word_token = tokenizer.encode(word)[0][1:-1]

            if r < 0.15 * 0.8:
                source.extend(len(word_token) * [tokenizer._token_mask_id])
                target.extend(word_token)
            elif r < 0.15 * 0.9:
                source.extend(word_token)
                target.extend(word_token)
            elif r < 0.15:
                source.extend([np.random.choice(tokenizer._vocab_size - 5) + 5 for _ in range(len(word_token))])
                target.extend(word_token)
            else:
                source.extend(word_token)
                target.extend([0] * len(word_token))

        # add end token
        source.append(tokenizer._token_end_id)
        target.append(0)

        if i == 0:
            segment = [0] * len(source)
        else:
            segment = [1] * len(source)

        sources.extend(source)
        targets.extend(target)
        segments.extend(segment)

    return sources, targets, segments


class pretrain_data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, = [], [], [], []

        for is_end, item in self.get_sample(shuffle):
            source_tokens, target_tokens, segment_ids = random_masking(item)

            is_masked = [0 if i == 0 else 1 for i in target_tokens]
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)
            batch_is_masked.append(is_masked)
            #             batch_nsp.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_target_ids = pad_sequences(batch_target_ids)
                batch_is_masked = pad_sequences(batch_is_masked)
                #                 batch_nsp = sequence_padding(batch_nsp)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked], None

                batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked = [], [], [], []


# 补齐最后一个batch
more_ids = batch_size - (len(pretrain_data) % batch_size)
pretrain_data = pretrain_data + pretrain_data[: more_ids]
pretrain_generator = pretrain_data_generator(data=pretrain_data, batch_size=batch_size)


# preln/postln/rezero
def get_base_model(version='pre'):

        return bert


def build_transformer_model_with_mlm(version='pre'):
    """带mlm的bert模型
    """
    assert version in ['pre', 'post', 'rezero']
    if version == 'rezero':
        attention_name = 'Transformer-%d-MultiHeadSelfAttention'
        feed_forward_name = 'Transformer-%d-FeedForward'
        skip_weights = []
        for i in range(12):
            skip_weights.append(feed_forward_name % i + '-Norm')
            skip_weights.append(feed_forward_name % i + '-ReWeight')
            skip_weights.append(attention_name % i + '-Norm')
            skip_weights.append(attention_name % i + '-ReWeight')

        bert = build_transformer_model(
            config_path,
            with_mlm='linear',
            model='rezero',
            return_keras_model=False,
            skip_weights_from_checkpoints=skip_weights,
            use_layernorm=None,
            reweight_trainable=True,
            init_reweight=0.,
        )
    else:
        bert = build_transformer_model(
            config_path,
            with_mlm='linear',
            model='rezero',
            return_keras_model=False,
            #         skip_weights_from_checkpoints=skip_weights,
            use_layernorm=version,
            reweight_trainable=False,
            init_reweight=1.,
        )

    proba = bert.model.output
    #     print(proba)
    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记

    #     nsp_label = Input(shape=(None, ), dtype='int64', name='nsp')  # nsp

    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def nsp_loss(inputs):
        """计算nsp loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        #         y_pred, _ = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        loss = K.mean(loss)
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        #         _, y_pred = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    def nsp_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        y_true = K.cast(y_true, K.floatx)
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.mean(acc)
        return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    #     nsp_loss = Lambda(nsp_loss, name='nsp_loss')([nsp_label, proba])
    #     nsp_acc = Lambda(nsp_acc, name='nsp_acc')([nsp_label, proba])

    train_model = Model(
        bert.model.inputs + [token_ids, is_masked], [mlm_loss, mlm_acc]
    )

    loss = {
            'mlm_loss': lambda y_true, y_pred: y_pred,
            'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
            #         'nsp_loss': lambda y_true, y_pred: y_pred,
            #         'nsp_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
        }

    return bert, train_model, loss


bert, train_model, loss = build_transformer_model_with_mlm(VERSION)

Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
Opt = extend_with_piecewise_linear_lr(Opt)

opt = Opt(learning_rate=pretrain_lr,
          exclude_from_weight_decay=['Norm', 'bias'],
          grad_accum_steps=2,
          lr_schedule={int(len(pretrain_generator) * pretrain_epochs * 0.1): 1.0,
                       len(pretrain_generator) * pretrain_epochs: 0},
          weight_decay_rate=0.01,
          )

train_model.compile(loss=loss, optimizer=opt)
# 如果传入权重，则加载。注：须在此处加载，才保证不报错。
if checkpoint_path is not None:
    bert.load_weights_from_checkpoint(checkpoint_path)

train_model.summary()

model_saved_path = './bert-wwm-model.ckpt'


class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self):
        self.loss = 1e6

    """自动保存最新模型
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']
        if (epoch + 1) % 10 == 0:
            bert.save_weights_as_checkpoint(model_saved_path+'-'+str(epoch+1))

        test_data = all_data[0][0]
        token_ids, segment_ids = tokenizer.encode(test_data)
        token_ids[9] = token_ids[10] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])
        print(tokenizer.decode(probs[0, 9:11].argmax(axis=1)), test_data)


# fine-tune data generator
class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_desc) in self.get_sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([int(label)])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(data=train_data, batch_size=batch_size)
val_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()

    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(val_generator)
        if acc > self.best_acc:
            self.best_acc = acc
        if (epoch + 1)% 10 == 0:
            bert.save_weights_as_checkpoint('bert.ckpt-{}'.format((epoch+1)%10))
        print('acc: {}, best acc: {}'.format(acc, self.best_acc))


if __name__ == '__main__':
    # pretrain bert use task data
    # 保存模型
    checkpoint = ModelCheckpoint()
    # 记录日志
    csv_logger = keras.callbacks.CSVLogger('training.log')

    train_model.fit(
        pretrain_generator.generator(),
        steps_per_epoch=len(pretrain_generator),
        epochs=pretrain_epochs,
        callbacks=[checkpoint, csv_logger],
    )

    # build task fine-tune model
    # reload weights without mlm
    # bert_without_mlm = build_transformer_model(checkpoint_path=model_saved_path,
    #                                            config_path=config_path, with_mlm=False)

    idx = 11
    feed_forward_name = 'Transformer-%d-FeedForward' % idx
    bert_without_mlm = bert.layers[feed_forward_name]
    output = Lambda(lambda x: x[:, 0])(bert_without_mlm.output)
    output = Dense(num_classes, activation='softmax')(output)

    model = Model(bert.inputs, output)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(fine_tune_lr),
                  metrics=['acc'])

    evaluator = Evaluator()
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=fine_tune_epochs,
                        callbacks=[evaluator])