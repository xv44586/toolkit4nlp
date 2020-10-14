#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :2020/06/01
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com

import os

os.environ['TF_KERAS'] = '1'

import tensorflow as tf
from toolkit4nlp.backend import K, keras

from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Model, Sequential
from toolkit4nlp.optimizers import extend_with_piecewise_linear_lr
from keras.callbacks import Callback

import numpy as np

x = np.arange(4).reshape((2, 2))
x = x * 1.0
y = x * 2.0
print(x, y)
model1 = Sequential([Dense(2, use_bias=False, kernel_initializer='ones')])
model2 = Sequential([Dense(2, use_bias=False, kernel_initializer='ones')])
model3 = Sequential([Dense(2, use_bias=False, kernel_initializer='ones')])
model1.compile(loss='mse', optimizer=SGD(lr=1))
model2.compile(loss='mse', optimizer=SGD(lr=1))


class PrintWeight(Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print('batch: ', batch)
        print('model: ', self.model.name)
        print(self.model.get_weights())


new_opt = extend_with_piecewise_linear_lr(SGD)
opt = new_opt(lr_schedule={10: 0.5}, lr=1)
model3.compile(loss='mse', optimizer=opt)

print_weight = PrintWeight()
model1.fit(x, y, epochs=10, batch_size=2, callbacks=[print_weight])
model2.fit(x, y, epochs=10, batch_size=2, callbacks=[print_weight])
model3.fit(x, y, epochs=10, batch_size=2, callbacks=[print_weight])
print(model1.get_weights())
print(model2.get_weights())
print(model3.get_weights())
