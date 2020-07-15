# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : models.py
import six

import tensorflow as tf
from keras.layers import *
from keras import initializers, activations

from .backend import K, keras, sequence_masking


class Transformer(object):
    """

    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 intermidial_act_fc,
                 hidden_dropout_prob,
                 attentioin_probs_dropout_prob,
                 initializer_range):
        '''

        :param vocab_size:
        :param hidden_size:
        :param num_hidden_layers:
        :param num_attention_heads:
        :param intermediate_size:
        :param intermidial_act_fc:
        :param hidden_dropout_prob:
        :param attentioin_probs_dropout_prob:
        :param initializer_range:
        '''
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError('The hidden size {hidden_size} is not a multiple of the number of attention heads {'
                             'num_attention_heads}')
        self.intermediate_size = intermediate_size
        self.intermediate_act_fc = intermidial_act_fc
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attentioin_probs_dropout_prob
        self.initializer = self.get_initializer(initializer_range)


# def get_activation(activation_string):
#     if not activation_string:
#         return None
#     if not isinstance(activation_string, six.string_types):
#         return activation_string
#
#     activation_string = activation_string.lower()
#     if activation_string == 'linear':
#         return None
#     if activation_string == 'gelu':
#         return gelu
#     elif activation_string == 'tanh':
#         return tf.nn.tanh
#     elif activation_string == 'relu':
#         return tf.nn.relu
#     elif activation_string == 'softmax':
#         return tf.nn.softmax
#     else:
#         raise ValueError('Unsupported activation {}'.format(activation_string))


def get_initializer(initializer_range):
    """
    截断正态分布初始化
    """
    return keras.initializers.TruncatedNormal(stddev=initializer_range)
