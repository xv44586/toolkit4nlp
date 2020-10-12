#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     :{DATE}
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com

import tensorflow as tf

from toolkit4nlp.backend import K, keras
from toolkit4nlp.backend import sequence_masking
from toolkit4nlp import initializers
from keras.layers import *
from keras import activations


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
        """
        多头注意力
        :param inputs: [q, k, v, a_mask, position_bias]
        :param mask: [q_mask, v_mask],
            q_mask 对query序列进行mask，针对padding；v_mask对value序列进行mask，防止看到某些位置value，如padding
        :param a_mask: Boolean，是否对attention进行mask
        :param position_bias: type of position bias， 使用指定类型的位置编码对attention里的位置进行偏移
        :return:
        """
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
        att = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if position_bias == 'relative':
            position_embeddings = inputs[idx]
            att = att + tf.einsum('bjhd,jkd->bhjk', qw, position_embeddings)

        if self.attention_scale:
            att = att / self.key_size ** 0.5

        # value mask
        att = sequence_masking(att, v_mask, 'add', -1)
        # attention mask
        if a_mask is not None:
            att = att - (1 - a_mask) * 1e12

        att = K.softmax(att)
        output = tf.einsum('bhjk,bkhd->bjhd', att, vw)
        # 继续处理位置编码
        if position_bias == 'relative':
            output = output + tf.einsum('bhjk,jkd->bjhd', att, position_embeddings)
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
                       'kernel_initializer': initializers.serialize(self.kernel_initializer)})
        return config


class LayerNormalization(Layer):
    """
    [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
    x_mean = mean(x)
    std = sqrt(mean(square(x - x_mean)) + epsilon)
    x = beta + (x - x_mean / std) * gamma
    """

    def __init__(self, center=True, scale=True, epsilon=None, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.center:
            self.beta = self.add_weight(shape=(input_shape[-1],), initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=(input_shape[-1],), initializer='ones', name='gamma')

    def call(self, inputs):
        output = inputs
        if self.center:
            mean = K.mean(inputs, axis=-1, keepdims=True)
            output = output - mean

        if self.scale:
            var = K.mean(K.square(output), axis=-1, keepdims=True)
            std = K.sqrt(var + self.epsilon)
            output = output / std
            output = output * self.gamma

        if self.center:
            output = output + self.beta

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
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen

    def build(self, input_shape):
        super(TokenAndPositionEmbedding, self).build(input_shape)
        self.token_emb = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, name='token_emb')
        self.pos_emb = Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, name='pos_emb')

    def call(self, inputs):
        maxlen = K.shape(inputs)[-1]
        token_emb = self.token_emb(inputs)
        pos = tf.range(start=0, limit=maxlen, delta=1)
        pos_emb = self.pos_emb(pos)
        return token_emb + pos_emb

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape) + [self.embed_dim])

    def get_config(self):
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        config = {
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim
        }
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """
    将位置Embedding与inputs 进行merge, merge mode : [add, concat].
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 embeddings_initializer='zero',
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='position_embedding',
                                          shape=(self.input_dim, self.output_dim),
                                          initializer=self.embeddings_initializer)

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
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'merge_mode': self.merge_mode,
                  'embeddings_initializer': initializers.serialize(self.embeddings_initializer)}
        return dict(list(base_config.items()) + list(config.items()))


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
        self.dense_2 = Dense(units=input_shape[-1],
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        base_config = super(FeedForward, self).get_config()
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        return dict(list(base_config.items()) + list(config.items()))


class BiasAdd(Layer):
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        self.bias = self.add_weight('bias',
                                    shape=(input_shape[-1],),
                                    initializer='zero',
                                    trainable=True)

    def call(self, inputs, **kwargs):
        return K.bias_add(inputs, self.bias)


class Embedding(keras.layers.Embedding):

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

        kernel = K.transpose(self.embeddings)
        return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        if self._mode == 'embedding':
            return super(Embedding, self).compute_output_shape(input_shape)

        return input_shape[:2] + (K.int_shape(self.embeddings)[0],)


class AttentionPooling1D(Layer):
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim

    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if not self.h_dim:
            self.h_dim = input_shape[-1]

        self.k_dense = Dense(self.h_dim, use_bias=False, activation='tanh')
        self.o_dense = Dense(1, use_bias=False)

    def call(self, x, mask=None):
        x0 = x
        x = self.k_dense(x0)
        x = self.o_dense(x)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        x = K.sum(x0 * x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        base_config = super(AttentionPooling1D, self).get_config()
        config = {'h_dim': self.h_dim}
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbedding(Layer):
    """相对位置编码
    ref：[Self-Attention with Relative Position Representations](http://arxiv.org/abs/1803.02155）
    """

    def __init__(self, input_dim, output_dim, embedding_initializer='zeros', **kwargs):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_initializer = initializers.get(embedding_initializer)

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='relativePositionEmbedding',
                                          shape=(self.input_dim, self.output_dim),
                                          initializer=self.embedding_initializer, )

    def call(self, inputs):
        relative_position_idx = self.compute_position_idx(inputs)
        return K.gather(self.embeddings, relative_position_idx)

    def compute_position_idx(self, inputs):
        q, v = inputs
        q_idx = K.arange(0, K.shape(q)[1], dtype='int32')
        q_idx = K.expand_dims(q_idx, 1)
        v_idx = K.arange(0, K.shape(v)[1], dtype='int32')
        v_idx = K.expand_dims(v_idx, 0)
        # 相对位置差
        position_idx = v_idx - q_idx
        max_position = (self.input_dim - 1) // 2
        position_idx = K.clip(position_idx, -max_position, max_position)
        position_idx = position_idx + max_position
        return position_idx

    def compute_output_shape(self, input_shape):
        return (None, None, self.output_dim)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def get_config(self):
        base_config = super(RelativePositionEmbedding, self).get_config()
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embedding_initializer': initializers.serialize(self.embedding_initializer)
        }
        return dict(list(base_config.items()) + list(config.items()))


class DGCNN(Layer):
    """
    膨胀卷积网络，优势是本质是一个CNN，所以速度上比RNNs快，同时通过不同的膨胀系数，如【1，3，5，8】可以来整合全局信息，
    此外，与残差网络结合，解决梯度消失问题，并让信息在多个通道流通。所以在处理序列数据时可以抛弃RNNs而尝试使用该结构。
    ref: https://spaces.ac.cn/archives/5409
    """

    def __init__(self, o_dim=None, k_size=3, dilation_rate=1, skip_connection=True, dropout_rate=None, **kwargs):
        super(DGCNN, self).__init__(**kwargs)

        self.o_dim = o_dim
        self.k_size = k_size
        self.dilation_rate = dilation_rate
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        super(DGCNN, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.dilation_rate,
            padding='same',
            name='dgcnn_conv1d'
        )
        if self.skip_connection and self.o_dim != input_shape[-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)

    def call(self, x, mask=None):
        x0 = x
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
        #         x = x0 * mask if mask is not None else x0
        x0 = Lambda(lambda x_: x_, output_shape=lambda s: s)(x0)  # drop mask so do not put mask to conv1d
        x = self.conv1d(x0)
        x, g = x[:, :, :self.o_dim], x[:, :, self.o_dim:]
        if self.dropout_rate is not None:
            g = K.in_train_phase(K.dropout(g, self.dropout_rate), g)
        g = K.sigmoid(g)
        # mask is none
        mask = mask if mask is not None else K.ones_like(x)

        if self.skip_connection:
            if K.int_shape(x0)[-1] != self.o_dim:
                x0 = self.conv1d_1x1(x0)
            return (x0 * (1 - g) + x * g) * mask
        return x * g * mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.o_dim,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self):
        base_config = super(DGCNN, self).get_config()
        config = {
            'o_dim': self.o_dim,
            'k_size': self.k_size,
            'dilation_rate': self.dilation_rate,
            'skip_connection': self.skip_connection,
            'dropout_rate': self.dropout_rate
        }
        return dict(list(base_config.items()) + list(config.items()))


class SinCosPositionEmbedding(Layer):
    def __init__(self, embedding_dim=None, method='add', embedding=None, **kwargs):
        self.method = method  # add or concate
        self.embedding_dim = embedding_dim  # encoder embedding dim, d_pos
        self.embedding = embedding
        super(SinCosPositionEmbedding, self).__init__(**kwargs)

    def call(self, inputs):
        #     PE_2i(p) = sin(p/10000^(2i/d_pos))
        #     PE_2i+1(p) = cos(p/10000^(2i/d_pos))
        batch_size, seq_len, word_emb_dim = K.shape(inputs)[0], K.shape(inputs)[1], K.shape(inputs)[2]
        if not self.embedding_dim or self.method == 'add':
            self.embedding_dim = word_emb_dim
        t = 2 * K.arange(self.embedding_dim / 2, dtype='float32') / K.cast(self.embedding_dim, dtype='float32')
        embedding_wise_pos = 1. / K.pow(10000., t)  # 1/10000 ^(2i/d_pos) , shape = (p_dim/2, )
        embedding_wise_pos = K.expand_dims(embedding_wise_pos, 0)  # (1, p_dim/2)
        word_wise_pos = K.cumsum(K.ones_like(inputs[:, :, 0]), axis=1)  # shape = [batch_size, seq_len]
        word_wise_pos = K.expand_dims(word_wise_pos, 2)  # (batch_size, seq_len, 1)
        position_embedding = K.dot(word_wise_pos, embedding_wise_pos)  # (batch_size, seq_len, p_dim/2)

        position_embedding = K.expand_dims(position_embedding, 3)
        position_embedding = K.reshape(K.concatenate([K.sin(position_embedding), K.cos(position_embedding)], axis=-1),
                                       shape=(batch_size, seq_len, -1))

        if self.method == 'add':
            return inputs + position_embedding

        return K.concatenate([inputs, position_embedding], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.method == 'add':
            return input_shape
        return input_shape[:-1] + (input_shape[-1] + self.embedding_dim,)

    def compute_mask(self, inputs, mask=None):
        return mask


class ConditionalRandomField(Layer):
    """
    带有可调节学习率的CRF，其中调节学习率可以参考[](https://kexue.fm/archives/6418) [](https://kexue.fm/archives/7196)
    sgd优化器下等效放大 lr_multiplier ** 2 , adam下等效放大 lr_multiplier 倍
    """

    def __init__(self, lr_multiplier=1, *args, **kwargs):
        super(ConditionalRandomField, self).__init__(*args, **kwargs)
        self.lr_multiplier = lr_multiplier  # 放大倍数

    def build(self, input_shape):
        super(ConditionalRandomField, self).build(input_shape)
        label_size = input_shape[-1]
        self._trans = self.add_weight(name='crf_trans',
                                      shape=(label_size, label_size),
                                      initializer='glorot_uniform',
                                      trainable=True)
        if self.lr_multiplier != 1:
            K.set_value(self._trans, K.eval(self._trans) / self.lr_multiplier)

    @property
    def trans(self):
        """权重按比例缩放回来"""
        if self.lr_multiplier != 1:
            return self.lr_multiplier * self._trans

        return self._trans

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        # 只是计算loss，并不改变输入
        if mask is not None:
            mask = K.cast(mask, K.floatx())

        return sequence_masking(inputs, mask, 1, 1)

    def path_score(self, inputs, labels):
        """
        :param inputs: (batch_size, timesteps, num_label), obtained from rnn(lstm, bilstm. etc.)
        :param labels: one-hot, (batch_size, timesteps, num_label) , real target series
        :return:  path score
        """
        point_score = tf.einsum('btn,btn->b', inputs, labels)  # 逐标签得分
        trans_score = tf.einsum('bti,ij,btj->b', labels[:, :-1], self.trans, labels[:, 1:])  # 标签转移得分
        return point_score + trans_score

    def log_norm_step(self, inputs, states):
        """递归求解归一化因子"""
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # batch_size,output_dim, 1
        trans = K.expand_dims(self.trans, 0)  # 1, output_dim, output_dim
        outputs = K.logsumexp(states + trans, 1)
        outputs += inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def dense_loss(self, y_true, y_pred):
        """y_true需要是one hot形式
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2, keepdims=True)
        mask = K.cast(mask, K.floatx())
        # 计算目标分数
        y_true, y_pred = y_true * mask, y_pred * mask
        target_score = self.path_score(y_pred, y_true)
        # 递归计算log Z
        init_states = [y_pred[:, 0]]
        y_pred = K.concatenate([y_pred, mask], axis=2)
        input_length = K.int_shape(y_pred[:, 1:])[1]
        log_norm, _, _ = K.rnn(
            self.log_norm_step,
            y_pred[:, 1:],
            init_states,
            input_length=input_length
        )  # 最后一步的log Z向量
        log_norm = K.logsumexp(log_norm, 1)  # logsumexp得标量
        # 计算损失 -log p
        return log_norm - target_score

    def sparse_loss(self, y_true, y_pred):
        """y_true需要是整数形式（非one hot）
        """
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 转为one hot
        y_true = K.one_hot(y_true, K.shape(self.trans)[0])
        return self.dense_loss(y_true, y_pred)

    def dense_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是one hot形式
        """
        y_true = K.argmax(y_true, 2)
        return self.sparse_accuracy(y_true, y_pred)

    def sparse_accuracy(self, y_true, y_pred):
        """训练过程中显示逐帧准确率的函数，排除了mask的影响
        此处y_true需要是整数形式（非one hot）
        """
        # 导出mask并转换数据类型
        mask = K.all(K.greater(y_pred, -1e6), axis=2)
        mask = K.cast(mask, K.floatx())
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 逐标签取最大来粗略评测训练效果
        y_pred = K.cast(K.argmax(y_pred, 2), 'int32')
        isequal = K.cast(K.equal(y_true, y_pred), K.floatx())
        return K.sum(isequal * mask) / K.sum(mask)

    def get_config(self):
        config = {
            'lr_multiplier': self.lr_multiplier,
        }
        base_config = super(ConditionalRandomField, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Loss(Layer):
    """
    特殊层，用于编写灵活loss
    """

    def __init__(self, output_idx=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_idx = output_idx  # 输出为inputs的idx部分

    def compute_loss(self, inputs):
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        loss = self.compute_loss(inputs)
        self.add_loss(loss)

        if self.output_idx is None:
            return inputs

        if type(self.output_idx) == list:
            return [inputs[idx] for idx in self.output_idx]
        return inputs[self.output_idx]

    def compute_output_shape(self, input_shape):
        if self.output_idx is None:
            return input_shape
        if type(self.output_idx) == list:
            return [input_shape[idx] for idx in self.output_idx]
        return input_shape[self.output_idx]

    def get_config(self):
        config = {'output_idx': self.output_idx}
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'Embedding': Embedding,
    'BiasAdd': BiasAdd,
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    'PositionEmbedding': PositionEmbedding,
    'FeedForward': FeedForward,
    'AttentionPooling1D': AttentionPooling1D,
    'ConditionalRandomField': ConditionalRandomField,
    'DGCNN': DGCNN,
    'SinCosPositionEmbedding': SinCosPositionEmbedding,
    'Loss': Loss,
    'RelativePositionEmbedding': RelativePositionEmbedding
}

keras.utils.get_custom_objects().update(custom_objects)
