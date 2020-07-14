#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com

import tensorflow as tf

from toolkit4nlp.backend import K, keras
from toolkit4nlp.backend import sequence_masking
from keras.layers import *
from keras import initializers


class MultiHeadAttention(Layer):
    """

    """

    def __init__(self, head_nums, head_size, key_size=None, use_bias=True, attention_scale=True,
                 kernel_initializer='glorot_uniform', **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_nums = head_nums
        self.head_size = head_size
        self.key_size = key_size or head_size
        self.output_dim = head_nums * head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.head_nums,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.head_nums,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

        self.combine_dense = Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, a_mask=None, position_bias=None):
        '''
        多头注意力
        TODO: position bias
        :param inputs: [q, k, v, a_mask, position_bias]
        :param mask: [q_mask, v_mask],
            q_mask 对query序列进行mask，针对padding；v_mask对value序列进行mask，防止看到某些位置value，如padding
        :param a_mask: Boolean，是否对attention进行mask
        :param position_bias: Boolean， 是否对attention里的位置进行偏移
        :return:
        '''
        q, k, v = inputs[:3]
        q_mask, v_mask, idx = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        if a_mask is not None:
            a_mask = inputs[idx]
            idx += 1

        # 投影变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        # 形状变换
        qw = K.reshape(qw, [-1, K.shape(q)[1], self.head_nums, self.key_size])
        kw = K.reshape(kw, [-1, K.shape(k)[1], self.head_nums, self.key_size])
        vw = K.reshape(vw, [-1, K.shape(v)[1], self.head_nums, self.head_size])

        # 计算attention
        # att = tf.einsum('bshk,bqhk->bhsq', qw, kw)
        att = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        if self.attention_scale:
            att = att / self.key_size ** 0.5

        # value mask
        att = sequence_masking(att, v_mask, 'add', -1)
        # attention mask
        if a_mask:
            att = att - (1 - a_mask) * 1e12

        att = K.softmax(att)

        # output = tf.einsum('bhsq,bqhk->bhsk', att, vw)
        output = tf.einsum('bhjk,bkhd->bjhd', att, vw)
        output = K.reshape(output, (-1, K.shape(output)[1], self.output_dim))
        output = self.combine_dense(output)
        # query mask
        output = sequence_masking(output, q_mask, 'mul')
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({'head_nums': self.head_nums,
                       'head_size': self.head_size,
                       'key_size': self.key_size,
                       'use_bias': self.use_bias,
                       'attention_scale': self.attention_scale,
                       'kernel_initializer': self.kernel_initializer})
        return config


class LayerNormalization(Layer):
    """
    [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    x = beta + (x - mean / std) * gamma
    """

    def __init__(self, center, scale, epsilon, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.center:
            self.beta = self.add_weight(shape=input_shape[-1], initializer='zero', name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=input_shape[-1], initializer='one', name='gamma')

    def call(self, inputs, **kwargs):
        output = inputs
        if self.center:
            mean = K.mean(inputs, axis=-1, keepdims=True)
            output = output - mean

        if self.scale:
            var = K.mean(K.square(inputs), axis=-1, keepdims=True)
            std = K.sqrt(var + self.epsilon)
            output = output / std * self.gamma

        if self.center:
            output += self.beta

        return output

    def get_config(self):
        base_config = super(LayerNormalization, self).get_config()
        base_config.update({"center": self.center,
                            "scale": self.scale,
                            "epsilon": self.epsilon})
        return base_config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, vocab_size, embed_dim, maxlen, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, name='token_emb')
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, name='pos_emb')
        self.embed_dim = embed_dim

    def call(self, inputs):
        maxlen = K.shape(inputs)[-1]
        token_emb = self.token_emb(inputs)
        pos = tf.range(start=0, limit=maxlen, delta=1)
        pos_emb = self.pos_emb(pos)
        return token_emb + pos_emb

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape) + [self.embed_dim])
