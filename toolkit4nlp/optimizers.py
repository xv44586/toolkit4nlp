# -*- coding: utf-8 -*-
# @Date    : 2020/7/6
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : optimizers.py
import numpy as np

import tensorflow as tf
from toolkit4nlp.backend import keras, K


class Adam(keras.optimizers.Optimizer):
    '''
    w_t = w_t-1 - update_t
    update_t = lr * m_t / sqrt(v_t)
    m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
    v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t**2
    由于更新前期梯度较小，容易朝着0方向走，所以通常加一个bias correct来校正方向
    m_t_hat = m_t / (1 + beta_1**t)
    v_t_hat = v_t / (1 + beta_2 ** t)

    ref:
    - [zhihu-zhuanlan](
            https://zhuanlan.zhihu.com/p/32230623)
    - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
    - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)

    '''
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-6, bias_correct=True, **kwargs):
        kwargs['name'] = kwargs.get('name', Adam)
        self.learning_rate = learning_rate
        super(Adam, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epsilon()
        self.bias_correct = bias_correct

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        var_dtype = var.dtype.base_type
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self._get_hyper(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self.get_slot('beta_2', var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_power = K.pow(beta_1_t, local_step)
        beta_2_power = K.pow(beta_2_t, local_step)

        # update
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * grad ** 2)
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(m, indices, (1 - beta_1_t) * grad)
                v_t = self._resource_scatter_add(v, indices, (1 - beta_2_t) * grad ** 2)

        #
        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correct:
                m_t = m_t / (1 + beta_1_power)
                v_t = v_t / (1 + beta_2_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var, indices):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learnint_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon
        }
        basic_config = super(Adam, self).get_config()
        return dict(list(basic_config.items()) + list(config.items()))
