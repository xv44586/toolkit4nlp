# -*- coding: utf-8 -*-
# @Date    : 2020/7/6
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : optimizers.py
import re

import numpy as np
import tensorflow as tf

from toolkit4nlp.backend import keras, K, is_tf_keras, piecewise_linear
from toolkit4nlp.utils import *


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
        kwargs['name'] = kwargs.get('name', 'Adam')
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
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
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

    def _resource_apply_dense(self, grad, var):
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


def export_to_custom_objects(extend_with_func):
    def new_extend_with_func(BaseOptimizer, name=None):
        NewOptimizer = extend_with_func(BaseOptimizer)
        if name:
            NewOptimizer.__name__ = name
        name = NewOptimizer.__name__
        keras.utils.get_custom_objects()[name] = NewOptimizer
        return NewOptimizer

    return new_extend_with_func


@export_to_custom_objects
def extend_with_gradient_accumulation_tf2(BaseOptimizer):
    class NewOptimizer(BaseOptimizer):
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            super(NewOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, 'gradient_accumulation')

        def _resource_apply(self, grad, var, indices=None):
            """interation % acc_steps==0 then update else accumulate
               思路是先判断是否累计了 acc_steps，如果没有，则update时保持原样，
               并累计当前梯度，否则，更新梯度并将累计的梯度置零
            """
            #  是否更新
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            #  获取梯度累计量
            gradient_accumulation = self.get_slot(var, 'gradient_accumulation')

            # 获取平均梯度
            gradient_t = gradient_accumulation / self.grad_accum_steps

            old_update = K.update

            # 根据条件判断是否真的更新
            def new_update(x, new_x):
                new_x = K.switch(cond, new_x, x)
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(gradient_t, var)
            K.update = old_update

            # 根据条件判断是否需要置零
            with tf.control_dependencies([op]):
                gradient_t = K.switch(cond, K.zeros_like(gradient_accumulation), gradient_accumulation)
                with tf.control_dependencies([K.update(gradient_accumulation, gradient_t)]):
                    if indices is None:
                        K.update(gradient_accumulation, gradient_accumulation + grad)
                    else:
                        self._resource_scatter_add(gradient_accumulation, indices, grad)
            return gradient_t

        def get_config(self):
            config = super(NewOptimizer, self).get_config()
            config.update({'grad_accum_steps': self.grad_accum_steps})
            return config

    return NewOptimizer


@export_to_custom_objects
def extend_with_gradient_accumulation(BaseOptimizer):
    """原生keras版"""

    class NewOptimizer(BaseOptimizer):
        @insert_arguments(grad_accum_steps=2)
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self._first_grad = True  # first grad

        @K.symbolic
        def get_updates(self, loss, params):
            # 是否更新
            cond = K.equal(self.iterations % self.grad_accum_steps, 0)
            cond = K.cast(cond, K.floatx())
            # 获取梯度
            grads = self.get_gradients(loss, params)
            self.accum_grads = [K.zeros(
                shape=K.int_shape(p), dtype=K.dtype(p), name='accum_grad_{}'.format(i)) for i, p in enumerate(params)]

            old_update = K.update

            def new_update(x, new_x):
                new_x = cond * new_x + (1 - cond) * x
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            # 累计更新
            with K.control_dependencies(updates):
                acc_updates = [
                    K.update(ag, g + (1 - cond) * ag) for ag, g in zip(self.accum_grads, grads)
                ]

            return acc_updates

        def get_gradients(self, loss, params):
            if self._first_grad:
                self._first_grad = False
                return super(NewOptimizer, self).get_gradients(loss, params)
            else:
                return [ag / self.grad_accum_steps for ag in self.accum_grads]

        def get_config(self):
            config = {'grad_accum_steps': self.grad_accum_steps}
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_weight_decay_tf2(BaseOptimizer):
    """增加权重衰减
    ref: [DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)
    大多数框架在实现L2 regularization时是使用weight decay，然而L2 regularization 与 weight decay 在标准 SGD下是等价的，
    但是当使用Adam时，缺不是等价的，原因是：
    g_t = ▽f_t-1 + λθ，其中λθ是 L2 loss的梯度
    m_t = β_1 * m_t-1 + (1 - β_1) * g_t
    v_t = β_2 * v_t-2 + (1 - β_2) * g_t^2
    θ_t = θ_t - 1 - α(m_t / v_t^0.5 + ε)
    代入上面三式后带有θ的项为 α（λθ/ v_t^0.5 + ε),这导致梯度变化越大的方向，权重约束越小，这显然不合理。
    L2 regularization应该是各向同性。一种改进这个问题的方法就是将梯度下降与weight decay 解耦，
    不在求梯度时代入weight decay ，而是在整个梯度下降完成后，加入weight decay，这样将梯度下降与weight decay解耦，
    达到L2 regularization效果
    """

    class NewOptimizer(BaseOptimizer):
        @insert_arguments(weight_decay_rate=0.01, exclude_from_weight_decay=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)

        def _resource_apply(self, grad, var, indices=None):
            old_update = K.update

            def new_update(x, new_x):
                if x is var and self._do_use_weight_decay(x):
                    lr_t = self._decayed_lr(x.dtype.base_dtype)
                    new_x = new_x - lr_t * self.weight_decay_rate * x
                return old_update(x, new_x)

            K.update = new_update
            op = super(NewOptimizer, self)._resource_apply(grad, var, indices)
            K.update = old_update
            return op

        def _do_use_weight_decay(self, param):
            """Whether to use L2 weight decay for `param_name`."""
            param_name = param.name
            if not self.weight_decay_rate:
                return False
            if self.exclude_from_weight_decay:
                for r in self.exclude_from_weight_decay:
                    if re.search(r, param_name) is not None:
                        return False
            return True

        def get_config(self):
            config = super(NewOptimizer, self).get_config()
            config.update({'weight_decay_rate': self.weight_decay_rate,
                           'exclude_from_weight_decay': self.exclude_from_weight_decay})
            return config

    return NewOptimizer


@export_to_custom_objects
def extend_with_weight_decay(BaseOptimizer):
    """原生keras版"""

    class NewOptimizer(BaseOptimizer):
        @insert_arguments(weight_decay_rate=0.01, exclude_from_weight_decay=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            if not hasattr(self, 'learning_rate'):
                self.learning_rate = self.lr

        @K.symbolic
        def get_update(self, loss, params):
            old_update = K.update

            def new_update(x, new_x):
                if x in params and self._do_use_weight_decay(x):
                    new_x = new_x - self.learning_rate * self.weight_decay_rate * x
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_update(loss, params)
            K.update = old_update
            return updates

        def _do_use_weight_decay(self, param):
            """Whether to use L2 weight decay for `param_name`."""
            param_name = param.name
            if not self.weight_decay_rate:
                return False
            if self.exclude_from_weight_decay:
                for r in self.exclude_from_weight_decay:
                    if re.search(r, param_name) is not None:
                        return False
            return True

        def get_config(self):
            config = {'weight_decay_rate': self.weight_decay_rate,
                      'exclude_from_weight_decay': self.exclude_from_weight_decay}
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


@export_to_custom_objects
def extend_with_piecewise_linear_lr_tf2(BaseOptimizer):
    """
    分段线性学习率，使用场景如 warmup
    """

    class NewOptimzer(BaseOptimizer):
        """
        schedule 是一个{ point: value} 的字典，如 {10: 1, 20: 0.5}代表从 0 到 10 步 lr 从 0 线性增加到 100% ，
        然后从 10 到 20 线性降低到 50%，之后一直保持 50% 不变
        """

        @insert_arguments(lr_schedule={0: 1})
        def __init__(self, *args, **kwargs):
            super(NewOptimzer, self).__init__(*args, **kwargs)
            self.lr_schedule = {int(t): v for t, v in self.lr_schedule.items()}

        def _decayed_lr(self, var_dtypes):
            """重写获取decayed learning rate 方法"""

            lr_t = super(NewOptimzer, self)._decayed_lr(var_dtypes)
            lr_rate = piecewise_linear(self.iterations, self.lr_schedule)
            return lr_t * K.cast(lr_rate, var_dtypes)

        def get_config(self):
            config = super(NewOptimzer, self).get_config()
            config.update({'lr_schedule': self.lr_schedule})
            return config

    return NewOptimzer


@export_to_custom_objects
def extend_with_piecewise_linear_lr(BaseOptimizer):
    """原生keras版"""
    class NewOptimizer(BaseOptimizer):
        @insert_arguments(lr_schedule={0:1})
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            self.lr_schedule = {int(t): v for t, v in self.lr_schedule.items()}

        @K.symbolic
        def get_update(self, loss, params):
            # 获取当前 step 的 lr rate
            lr_rate_t = piecewise_linear(self.iterations, self.lr_schedule)

            old_update = K.update

            def new_update(x, new_x):
                new_x = x + (new_x - x) * lr_rate_t  # 按照当前lr rate 缩放 update
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_update(loss, params)
            K.update = old_update
            return updates

        def get_config(self):
            config = {'lr_schedule': self.lr_schedule}
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer


# keras or tf.keras
if is_tf_keras:
    extend_with_piecewise_linear_lr = extend_with_piecewise_linear_lr_tf2
    extend_with_gradient_accumulation = extend_with_gradient_accumulation_tf2
    extend_with_weight_decay = extend_with_weight_decay_tf2
else:
    Adam = keras.optimizers.Adam

custom_objects = {
    'Adam': Adam
}

keras.utils.get_custom_objects().update(custom_objects)
