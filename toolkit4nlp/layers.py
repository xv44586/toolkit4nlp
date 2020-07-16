#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com

import tensorflow as tf

from toolkit4nlp.backend import K, keras
from toolkit4nlp.backend import sequence_masking
from keras.layers import *
from keras import initializers, layers, activations


class Layer(keras.layers.Layer):
    # 所有自定义layer都支持masking
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True


class MultiHeadAttention(Layer):
    """
    多头注意力机制
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

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]

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
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )


    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalizationV2(Layer):
    """
    [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    x = beta + (x - mean / std) * gamma
    """

    def __init__(self, center=True, scale=True, epsilon=None, **kwargs):
        super(LayerNormalizationV2, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.center:
            self.beta = self.add_weight(shape=(input_shape[-1], ), initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=(input_shape[-1], ), initializer='ones', name='gamma')

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

    def compute_mask(self, inputs, mask=None):
        return mask

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


class PositionEmbedding(Layer):
    """
    将位置Embedding与inputs 进行merge, merge mode : [add, concat].
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 initializer='zero',
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='position_embedding',
                                          shape=(self.input_dim, self.output_dim),
                                          initializer=self.initializer)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]
        pos_embedding = self.embeddings[:seq_length]
        pos_embedding = K.expand_dims(pos_embedding, 0)
        if self.merge_mode != 'add':
            pos_embedding = K.tile(pos_embedding, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embedding
        return K.concatenate([inputs, pos_embedding], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape

        return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        base_config = super(PositionEmbedding, self).get_config()
        base_config.update({'input_dim': self.input_dim, 'output_dim': self.output_dim, 'merge_mode': self.merge_mode,
                            'initializer': initializers.serialize(self.initializer)})
        return base_config


class FeedForward(Layer):
    """
    Dense(units, activation) -> Dense(output_dim)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)

        self.dense_1 = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.dense_2 = Dense(units=input_shape[-1], use_bias=self.use_bias, kernel_initializer=self.kernel_initializer)

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


class BiadAdd(Layer):
    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=(input_shape[-1],), initializer='zero', trainable=True)

    def call(self, inputs, **kwargs):
        return K.bias_add(inputs, self.bias)


class Embedding(layers.Embedding):

    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if self._mode == 'embedding':
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """
        embedding mode: 普通embedding， dense mode: 无bias的Dense，x dot embedding.T
        :param inputs:
        :param mode:
        :return:
        """
        self._mode = mode
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        return K.dot(inputs, K.transpose(self.embeddings))

    def compute_output_shape(self, input_shape):
        if self._mode == 'embedding':
            return super(Embedding, self).compute_output_shape(input_shape)

        return input_shape[:2] + (K.int_shape(self.embeddings)[0],)
