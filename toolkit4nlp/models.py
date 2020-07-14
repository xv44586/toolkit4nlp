# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : models.py
import tensorflow as tf
from keras.layers import *
from keras import initializers, activations

from .backend import K, sequence_masking


class MultiHeadAttention(Layer):
    '''

    '''

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
        qw = K.reshape(qw, [-1, K.shape(inputs)[1], self.head_nums, self.key_size])
        kw = K.reshape(kw, [-1, K.shape(inputs)[1], self.head_nums, self.key_size])
        vw = K.reshape(vw, [-1, K.shape(inputs)[1], self.head_nums, self.head_size])

        # 计算attention
        att = tf.einsum('bshk,bqhk->bhsq', qw, kw)
        if self.attention_scale:
            att = att / self.key_size ** 0.5

        # value mask
        att = sequence_masking(att, v_mask, 'add', -1)
        # attention mask
        if a_mask:
            att = att - (1 - a_mask) * 1e12

        att = K.softmax(att)

        output = tf.einsum('bhsq,bqhk->bhsk', att, vw)
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

