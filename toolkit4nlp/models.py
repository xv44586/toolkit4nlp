# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : models.py
import six
import json

import tensorflow as tf
from keras.layers import *
from keras import initializers, activations
from keras.models import Model

from toolkit4nlp.backend import K, keras, sequence_masking
from toolkit4nlp.layers import *


class Transformer(object):
    """
    base class
    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act=None,
                 hidden_dropout_prob=None,
                 attentioin_probs_dropout_prob=None,
                 initializer_range=None,
                 embedding_size=None,
                 seq_length=None,
                 attention_mask=None,
                 layers=None,
                 name=None,
                 **kwargs):
        """

        :param vocab_size:
        :param hidden_size:
        :param num_hidden_layers:
        :param num_attention_heads:
        :param intermediate_size:
        :param intermidial_act_fc:
        :param hidden_dropout_prob:
        :param attentioin_probs_dropout_prob:
        :param initializer_range:
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError('The hidden size {hidden_size} is not a multiple of the number of attention heads {'
                             'num_attention_heads}')
        self.num_attention_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attentioin_probs_dropout_prob or hidden_dropout_prob
        self.initializer = self.get_initializer(initializer_range)
        self.embedding_size = embedding_size or hidden_size
        self.seq_length = seq_length
        self.attention_mask = attention_mask
        self.layers = layers or {}
        self.name = name
        self.built = False

    def build(self, **kwargs):
        if self.built:
            return

        # inputs
        self.inputs = self.get_inputs()

        # call
        self.outputs = self.call(self.inputs)

        # model
        self.model = Model(self.inputs, self.outputs, name=self.name)

        self.built = True

    def call(self, inputs):
        # embedding
        outputs = self.apply_embeddings(inputs)

        # main transformer layers
        for _ in range(self.num_hidden_layers):
            outputs = self.apply_transformer(outputs, _)

        # task related layers
        outputs = self.apply_task_related(outputs)
        return outputs

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_transformer(self, inputs, idx):
        raise NotImplementedError

    def apply_task_related(self, inputs):
        raise NotImplementedError

    def apply(self, inputs, layer=None, layer_name=None, arguments=None, **kwargs):
        '''
         记录layer信息方便后续mapping权重服务；重用同名layer;
         layer(name=layer_name, **kwargs)(inputs, **arguments)
        :param layer_name:
        :param inputs:
        :param layer: 具体layer
        :param arguments: 计算时使用参数
        :param kwargs: 初始化参数
        :return:
        '''
        if layer is Dropout and self.hidden_dropout_prob == 0:
            return inputs

        arguments = {} if arguments is None else arguments
        if layer_name not in self.layers:
            current_layer = layer(name=layer_name, **kwargs)
            self.layers[layer_name] = current_layer

        return self.layers[layer_name](inputs, **arguments)

    def get_initializer(self, initializer_range):
        """
        截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=initializer_range)

    def get_inputs(self):
        raise NotImplementedError

    def compute_attention_mask(self, inputs):
        return self.attention_mask

    def load_variable(self, checkpoint, name):
        return tf.train.load_variable(checkpoint, name)

    def variable_mapping(self):
        # keras 层与checkpoint变量直接的映射关系
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        mapping = mapping or self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weights_values_pairs = []
        for layer_name, variables in mapping.items():
            weights = self.layers[layer_name].trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]

            weights_values_pairs.extend(zip(weights, values))

        K.batch_set_value(weights_values_pairs)

    def create_variable(self, name, value):
        return tf.Variable(name=name, initial_value=value)

    def save_weights_as_checkpoint(self, checkpoint_path, mapping=None):
        mapping = mapping or self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            for layer_name, variables in mapping.items():
                layer = self.layers[layer_name]
                weights = K.batch_get_value(layer.trainable_weights)
                for variable, weight in zip(variables, weights):
                    self.create_variable(variable, weight)
            with tf.Session() as sess:
                sess.run(tf.global_variable_initialzer())
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path, write_meta_graph=False)


class BERT(Transformer):
    """

    """

    def __init__(self,
                 max_position,  # 序列最大长度
                 with_pool=False,  # 是否包含pooler部分
                 with_nsp=False,  # 是否包含NSP部分
                 with_mlm=False,  # 是否包含mlm部分
                 **kwargs
                 ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position

        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        # nsp need pooler
        if with_nsp and not with_pool:
            self.with_pool = True

    def get_inputs(self):
        token_in = Input((self.seq_length,), name='Input-Token')
        segment_in = Input((self.seq_length,), name='Input-Segment')

        return [token_in, segment_in]

    def apply_embeddings(self, inputs):
        """token_embedding + segment_embedding + position_embedding"""
        x, s = inputs[:2]
        token_embedding = self.apply(inputs=x,
                                     layer=Embedding,
                                     layer_name='Embedding-Token',
                                     input_dim=self.vocab_size,
                                     output_dim=self.embedding_size,
                                     embeddings_initializer=self.initializer,
                                     mask_zero=True
                                     )
        segment_embedding = self.apply(s,
                                       Embedding,
                                       layer_name='Embedding-Segment',
                                       input_dim=2,
                                       output_dim=self.embedding_size,
                                       embeddings_initializer=self.initializer,
                                       )
        token_with_seg = self.apply([token_embedding, segment_embedding], Add, 'Embedding-Token-Segment')
        x = self.apply(token_with_seg,
                       PositionEmbedding, 'Embedding-Position',
                       input_dim=self.max_position,
                       output_dim=self.embedding_size,
                       initializer=self.initializer,
                       merge_mode='add')

        x = self.apply(x,
                       LayerNormalization, 'Embedding-Norm')
        x = self.apply(x, Dropout,
                       'Embedding-Dropout', rate=self.hidden_dropout_prob)
        if self.hidden_size != self.embedding_size:
            x = self.apply(x, Dense, 'Embedding-Mapping',
                           units=self.hidden_size, kernel_initializer=self.initializer)

        return x

    def apply_transformer(self, inputs, idx):
        """
        Att --> Dropout --> Add --> LN --> FFN --> Dropout -->  Add --> LN
        :param idx:
        :param inputs:
        :return:
        """
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % idx
        feed_forward_name = 'Transformer-%d-FeedForward' % idx
        attention_mask = self.compute_attention_mask(idx)

        x_pre, x = inputs, [inputs, inputs, inputs]
        arguments = {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = attention_mask

        # self-attention
        x = self.apply(x,
                       MultiHeadAttention,
                       attention_name,
                       head_nums=self.num_attention_heads,
                       head_size=self.num_attention_size,
                       kernel_initializer=self.initializer)
        x = self.apply(x,
                       Dropout,
                       '%s-Dropout' % attention_name,
                       rate=self.attention_probs_dropout_prob)

        x = self.apply([x_pre, x],
                       Add,
                       '%s-Add' % attention_name
                       )

        x = self.apply(x,
                       LayerNormalization,
                       '%s-Norm' % attention_name,
                       )

        # feedforward
        x_pre = x
        x = self.apply(x,
                       FeedForward,
                       feed_forward_name,
                       units=self.intermediate_size,
                       activation=self.hidden_act,
                       kernel_initializer=self.initializer
                       )
        x = self.apply(x,
                       Dropout,
                       '%s-Dropout' % feed_forward_name,
                       rate=self.hidden_dropout_prob)
        x = self.apply([x_pre, x],
                       Add,
                       '%s-Add' % feed_forward_name)
        x = self.apply(x, LayerNormalization, '%s-Norm' % feed_forward_name)
        return x

    def apply_task_related(self, inputs):
        """
        跟据不同task加不同的layer产生不同的outputs
        :param inputs:
        :return:
        """
        x = inputs
        outputs = [x]
        if self.with_pool:
            # pooler 提取cls向量
            x = self.apply(x, layer=Lambda, layer_name='Pooler', function=lambda x: x[:, 0])
            x = self.apply(x, Dense,
                           'Pooler-Dense',
                           units=self.hidden_size,
                           activation='tanh',
                           kernel_initializer=self.initializer)
            if self.with_nsp:
                # Next sentence prediction
                x = self.apply(x, Dense, 'NSP-Proba', units=2, activation='softmax',
                               kernel_initializer=self.initializer)

            outputs.append(x)

        if self.with_mlm:
            # Mask language model, Dense --> Norm --> Embedding --> Biasadd --> softmax
            x = outputs[0]
            x = self.apply(x, Dense, 'MLM-Dense', units=self.embedding_size,
                           activation=self.hidden_act, kernel_initializer=self.initializer)
            x = self.apply(x,
                           LayerNormalization,
                           'MLM-Norm',
                           )
            # 重用embedding-token layer
            x = self.apply(x, Embedding, 'Embedding-Token', arguments={'mode': 'dense'})
            x = self.apply(x, BiadAdd, 'MLM-Bias')
            x = self.apply(x, Activation, 'MLM-Activation', activation='softmax')
            outputs.append(x)

        if len(outputs) == 1:
            return outputs[0]
        if len(outputs) == 2:
            return outputs[1]

        return outputs[1:]

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        # if name in [
        #     'bert/embeddings/word_embeddings',
        #     'cls/predictions/output_bias',
        # ]:
        #     if self.keep_tokens is None:
        #         return variable
        #     else:
        #         return variable[self.keep_tokens]
        if name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        return_keras_model=True,
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')

    models = {
        'bert': BERT,
    }

    if isinstance(model, six.string_types):
        model = model.lower()
        MODEL = models[model]
    else:
        MODEL = model

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer
