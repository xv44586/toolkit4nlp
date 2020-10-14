# -*- coding: utf-8 -*-
# @Date    : 2020/7/21
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : pretraining.py
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import glob

import tensorflow as tf
from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.optimizers import Adam, extend_with_gradient_accumulation, extend_with_wight_decay
from toolkit4nlp.optimizers import extend_with_piecewise_linear_lr
from keras.models import Model
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
learning_rate = 0.00176
weight_decay_rate = 0.01
num_warmup_steps = 3125
num_train_steps = 125000
steps_per_epoch = 10000
grad_accum_steps = 16  # 大于1即表明使用梯度累积,即每 batch_size * grad_accum_steps 步才会更新一次
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']
# warm up and global learning_rate decay
lr_schedule = {
    num_warmup_steps * grad_accum_steps: 1.0,
    num_train_steps * grad_accum_steps: 0.0,
}

# load dataset
dataset = TrainingDataSetRoBERTa.load_tfrecord(record_names=file_names, seq_length=seq_length, batch_size=batch_size)

"""
使用RoBerta的方式训练，即取消NSP任务，只保留mask language model 任务
"""
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
"""
bert的优化器中的learning rate 策略分为二块：
首先是全局从init_learning_rate 线性 decay to 0 ;
其次，在最初的warmup steps 中，learning_rate 从 0 线性增长到 init_learning_rate;
优化器选择Adam，里面有两点需要注意：
首先， 除了 *-Norm 和 Bias 层，其他层都进行 weight_decay；
其次，Adam 中 没有使用 bias_correct
"""
optimizer = extend_with_piecewise_linear_lr(Adam)
optimizer = extend_with_wight_decay(optimizer)

params = {
    'lr_schedule': lr_schedule,
    'learning_rate': learning_rate,
    'weight_decay_rate': weight_decay_rate,
    'exclude_from_weight_decay': exclude_from_weight_decay,
    'bias_correct': False
}

# grad acc
if grad_accum_steps > 1:
    optimizer = extend_with_gradient_accumulation(optimizer)
    params['grad_accum_steps'] = grad_accum_steps

optimizer = optimizer(**params)

# compile
train_model.compile(optimizer=optimizer, loss=loss)

# load init ckpt
if ckpt is not None:
    bert.load_weights_from_checkpoint(ckpt)


# callback
class ModelCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(model_save_path, overwrite=True)


model_checkpoint = ModelCheckpoint()
csv_logger = keras.callbacks.CSVLogger('train.log')

# fit
train_model.fit(dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=[model_checkpoint, csv_logger])
