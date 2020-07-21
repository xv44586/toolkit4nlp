# -*- coding: utf-8 -*-
# @Date    : 2020/7/21
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : pretraining.py
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import glob

import tensorflow as tf
from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import  build_transformer_model
from keras.models import Model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.layers import Input, Lambda
from preprocess import TrainingDataSetRoBERTa

floatx = K.floatx()

config = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
ckpt = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

model_save_path = '../saved_model/bert_model.ckpt'

file_names = glob.glob('../corpus_record/*')

seq_length = 512
batch_size = 8
learning_rate = 1e-5
epochs = 5
steps_per_epoch = 100

# load dataset
dataset = TrainingDataSetRoBERTa.load_tfrecord(record_names=file_names, seq_length=seq_length, batch_size=batch_size)

# build model
bert = build_transformer_model(
    config, ckpt, with_mlm='linear', return_keras_model=False
)
proba = bert.model.output

# 辅助输入
token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
is_masked = Input(shape=(None,), dtype=floatx, name='is_masked')  # mask标记


def mlm_loss(inputs):
    """计算loss的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs
    loss = K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
    return loss


def mlm_acc(inputs):
    """计算准确率的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs
    y_true = K.cast(y_true, floatx)
    acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
    return acc


mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])

train_model = Model(
    bert.model.inputs + [token_ids, is_masked], [mlm_loss, mlm_acc]
)

loss = {
    'mlm_loss': lambda y_true, y_pred: y_pred,
    'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
}

# optimizer
optimizer = keras.optimizers.Adam(learning_rate)

# compile
train_model.compile(optimizer=optimizer, loss=loss)
if ckpt is not None:
    bert.load_weights_from_checkpoint(ckpt)


# callback
class ModelCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(model_save_path, overwrite=True)


model_checkpoint = ModelCheckpoint()
csv_logger = keras.callbacks.CSVLogger('train.log')

# fit
train_model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint, csv_logger])
