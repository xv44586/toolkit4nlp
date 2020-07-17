# -*- coding: utf-8 -*-
# @Date    : 2020/7/6
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : backend.py
# 判断是tf.keras还是纯keras的标记
import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf

is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K

    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K


def gelu_tanh(x):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def set_gelu_version(version_str):
    if version_str not in ['gelu_erf', 'gelu_tanh']:
        raise ValueError('{} version not supported.'.format(version_str))
    if version_str == 'gelu_erf':
        keras.utils.get_custom_objects().update({'gelu': gelu_erf()})

    keras.utils.get_custom_objects().update({'gelu': gelu_tanh()})


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
        mask = K.expand_dims(mask, 1)
    # right
    for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
        mask = K.expand_dims(mask, K.ndim(mask))

    if mode in [0, 'mul']:
        return x * mask

    return x - (1 - mask) * 1e12


custom_objects = {
    'gelu': gelu_erf,
}

keras.utils.get_custom_objects().update(custom_objects)
