#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com
import tensorflow as tf
import keras
from keras import layers
from toolkit4nlp.layers import *


vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,), name='inputs')
embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim)
x = embedding_layer(inputs)
xi = x
x = MultiHeadAttention(head_nums=num_heads, head_size=embed_dim // num_heads)([x, x, x])
x = Dropout(0.2)(x)
x = Add()([xi, x])
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = Dense(20, name='my_dense')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
