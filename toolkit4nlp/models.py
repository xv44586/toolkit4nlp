# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : models.py
import tensorflow as tf
from keras.layers import *
from keras import initializers, activations

from .backend import K, sequence_masking


