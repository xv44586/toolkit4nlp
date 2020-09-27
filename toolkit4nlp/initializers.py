# -*- coding: utf-8 -*-
# @Date    : 2020/9/27
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : initializers.py
from toolkit4nlp.backend import keras
from keras.initializers import *


class Sinusoidal(Initializer):
    """I

    """

    def __call__(self, shape, dtype=None):
        """
        直接使用Sin-Cos形式的位置编码
        """
        vocab_size, depth = shape
        embeddings = np.zeros(shape)
        for j in range(vocab_size):
            for k in range(depth // 2):
                theta = j / np.power(10000, 2. * k / depth)
                embeddings[j, 2 * k] = np.sin(theta)
                embeddings[j, 2 * k + 1] = np.cos(theta)

        return embeddings


sinusoidal = Sinusoidal

custom_objects = {
    'sinusoidal': sinusoidal,
    'Sinusoidal': Sinusoidal
}
keras.utils.get_custom_objects().update(custom_objects)
