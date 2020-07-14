# -*- coding: utf-8 -*-
# @Date    : 2020/7/6
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : backend.py
# 判断是tf.keras还是纯keras的标记
import os, sys
from distutils.util import strtobool
import numpy as np


is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K


def gelu(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (
            1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    )
    return x * cdf


def sequence_masking(x, mask, mode=0, axis=1):
    '''
    mask shape: [batch_size, seq_length]
    :param x:
    :param mask: 0,1 矩阵
    :param mode: 直接相乘 or 减大数模拟
    :param axis:
    :return:
    '''
    if mask is None or mode not in [0, 1, 'mul', 'add']:
        return x

    if axis == -1:
        axis = K.ndim(x) - 1
    assert axis > 0, 'axis must greater than 0'
    # 形状扩充
    # left
    for _ in range(axis - 1):
        mask = K.expand_dims(x, 1)
    # right
    for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
        mask = K.expand_dims(mask, K.ndim(mask))

    if mode in [0, 'mul']:
        return x * mask

    return x - (1 - mask) * 1e12
