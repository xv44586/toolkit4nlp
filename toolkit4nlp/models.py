# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : models.py
import six
import json
import numpy as np

from toolkit4nlp.layers import *
from keras.models import Model
from toolkit4nlp.utils import remove_arguments


class Transformer(object):
    """
    base class
    """

    def __init__(self,
                 vocab_size,  # 词表大小
                 hidden_size,  # 编码维度
                 num_hidden_layers,  # Transformer总的层数
                 num_attention_heads,  # Attention头数
                 intermediate_size,  # FeedForward 层的隐层维度
                 hidden_act=None,  # FeedFoward 层的激活函数
                 dropout_rate=None,  # dropout 比例
                 embedding_size=None,  # embedding 层维度
                 attention_key_size=None,  # attentioin 中 Q, K 的head size
                 sequence_length=None,  # 是否固定序列长度
                 attention_mask=None,  # attention 层mask
                 layers=None,  # 外部传人的layer
                 name=None,  # 模型名
                 prefix=None,  # layer name 的前缀
                 keep_tokens=None,  # 自定义保留token
                 **kwargs):
        """
        """
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError('The hidden size {hidden_size} is not a multiple of the number of attention heads {'
                             'num_attention_heads}')
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate or 0
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.attention_mask = attention_mask
        self.position_bias = None
        self.layers = layers or {}
        self.name = name
        self.prefix = prefix or ''
        self.keep_tokens = keep_tokens
        self.built = False

    def build(self, **kwargs):
        """构建模型
        """
        if self.built:
            return None

        # inputs
        inputs = self.get_inputs()
        self.set_inputs(inputs)
        # call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # model
        self.model = Model(self.inputs, self.outputs, name=self.name)

        self.built = True

    def call(self, inputs):
        """模型的计算图
        """
        # embedding
        outputs = self.apply_embeddings(inputs)

        # main transformer layers
        for _ in range(self.num_hidden_layers):
            outputs = self.apply_transformer_layers(outputs, _)

        # task related layers
        outputs = self.apply_task_related(outputs)
        return outputs

    def prefixed(self, name):
        """增加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, name=None, arguments=None, **kwargs):
        """
         记录layer信息方便后续mapping权重服务；重用同名layer;
         layer(name=layer_name, **kwargs)(inputs, **arguments)
        :param inputs: 上一层的输出
        :param layer: 具体layer
        :param name: 层的名字
        :param arguments: 计算时使用参数
        :param kwargs: 初始化参数
        :return:
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        arguments = {} if arguments is None else arguments

        # add prefix
        name = self.prefixed(name)
        kwargs['name'] = name

        if name not in self.layers:
            current_layer = layer(**kwargs)
            name = current_layer.name
            self.layers[name] = current_layer

        if inputs is None:
            return self.layers[name]

        return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_transformer_layers(self, inputs, idx):
        raise NotImplementedError

    def apply_task_related(self, inputs):
        raise NotImplementedError

    def compute_attention_mask(self, inputs):
        return self.attention_mask

    def set_inputs(self, inputs):
        """设置input 和 inputs
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output 和 outputs
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """
        截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def load_variable(self, checkpoint, name):
        return tf.train.load_variable(checkpoint, name)

    def load_embeddings(self, embeddings):
        if self.keep_tokens:
            return embeddings[self.keep_tokens]
        return embeddings

    def create_variable(self, name, value):
        return tf.Variable(initial_value=value, name=name)

    def variable_mapping(self):
        # keras 层与checkpoint变量直接的映射关系
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据 mapping 从 checkpoint 加载权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weights_values_pairs = []
        for layer_name, variables in mapping.items():
            weights = self.layers[layer_name].trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]

            weights_values_pairs.extend(zip(weights, values))

        K.batch_set_value(weights_values_pairs)

    def save_weights_as_checkpoint(self, checkpoint_path, mapping=None):
        """根据mapping 将权重保存为 checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            for layer_name, variables in mapping.items():
                layer = self.layers[layer_name]
                weights = K.batch_get_value(layer.trainable_weights)
                for variable, weight in zip(variables, weights):
                    self.create_variable(variable, weight)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path, write_meta_graph=False)


class BERT(Transformer):
    """
    Google Bert
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
        token_in = self.apply(layer=Input,
                              name='Input-Token',
                              shape=(self.sequence_length,))
        segment_in = self.apply(layer=Input,
                                name='Input-Segment',
                                shape=(self.sequence_length,))

        return [token_in, segment_in]

    def apply_embeddings(self, inputs):
        """token_embedding + segment_embedding + position_embedding
        """
        x, s = inputs[:2]
        token_embedding = self.apply(inputs=x,
                                     layer=Embedding,
                                     name='Embedding-Token',
                                     input_dim=self.vocab_size,
                                     output_dim=self.embedding_size,
                                     embeddings_initializer=self.initializer,
                                     mask_zero=True
                                     )
        segment_embedding = self.apply(s,
                                       Embedding,
                                       name='Embedding-Segment',
                                       input_dim=2,
                                       output_dim=self.embedding_size,
                                       embeddings_initializer=self.initializer,
                                       )
        token_with_seg = self.apply([token_embedding, segment_embedding], Add, name='Embedding-Token-Segment')
        x = self.apply(token_with_seg,
                       PositionEmbedding,
                       name='Embedding-Position',
                       input_dim=self.max_position,
                       output_dim=self.embedding_size,
                       embeddings_initializer=self.initializer,
                       merge_mode='add')

        x = self.apply(x,
                       LayerNormalization,
                       name='Embedding-Norm')
        x = self.apply(x,
                       Dropout,
                       name='Embedding-Dropout',
                       rate=self.dropout_rate)
        if self.hidden_size != self.embedding_size:
            x = self.apply(x,
                           Dense,
                           name='Embedding-Mapping',
                           units=self.hidden_size,
                           kernel_initializer=self.initializer)

        return x

    def apply_transformer_layers(self, inputs, idx):
        """
        Att --> Dropout --> Add --> LN --> FFN --> Dropout -->  Add --> LN
        """
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % idx
        feed_forward_name = 'Transformer-%d-FeedForward' % idx
        attention_mask = self.compute_attention_mask(idx)

        x_pre, x = inputs, [inputs, inputs, inputs]
        arguments = {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)

        # self-attention
        x = self.apply(x,
                       MultiHeadAttention,
                       name=attention_name,
                       head_nums=self.num_attention_heads,
                       head_size=self.attention_head_size,
                       arguments=arguments,
                       kernel_initializer=self.initializer)

        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % attention_name,
                       rate=self.dropout_rate)

        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % attention_name
                       )

        x = self.apply(x,
                       LayerNormalization,
                       name='%s-Norm' % attention_name,
                       )

        # feedforward
        x_pre = x
        x = self.apply(x,
                       FeedForward,
                       name=feed_forward_name,
                       units=self.intermediate_size,
                       activation=self.hidden_act,
                       kernel_initializer=self.initializer
                       )
        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % feed_forward_name,
                       rate=self.dropout_rate)
        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % feed_forward_name)
        x = self.apply(x, LayerNormalization, name='%s-Norm' % feed_forward_name)
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
            x = self.apply(x, layer=Lambda, name='Pooler', function=lambda x: x[:, 0])
            x = self.apply(x,
                           layer=Dense,
                           name='Pooler-Dense',
                           units=self.hidden_size,
                           activation='tanh',
                           kernel_initializer=self.initializer)
            if self.with_nsp:
                # Next sentence prediction
                x = self.apply(x,
                               layer=Dense,
                               name='NSP-Proba',
                               units=2,
                               activation='softmax',
                               kernel_initializer=self.initializer)

            outputs.append(x)

        if self.with_mlm:
            # Mask language model, Dense --> Norm --> Embedding --> Biasadd --> softmax
            x = outputs[0]
            x = self.apply(x,
                           layer=Dense,
                           name='MLM-Dense',
                           units=self.embedding_size,
                           activation=self.hidden_act,
                           kernel_initializer=self.initializer)
            x = self.apply(x,
                           LayerNormalization,
                           'MLM-Norm',
                           )
            # 重用embedding-token layer
            x = self.apply(x, Embedding, 'Embedding-Token', arguments={'mode': 'dense'})
            x = self.apply(x, BiasAdd, 'MLM-Bias')
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
        if name == 'cls/seq_relationship/output_weights':
            return variable.T
        elif name in ['bert/embeddings/word_embeddings',
                      'cls/predictions/output_bias', ]:
            return self.load_embeddings(variable)
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


class ELECTRA(BERT):
    @remove_arguments('with_mlm', 'with_pool')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping()
        mapping = {k: [name.replace('bert/', 'electra/') for name in v] for k, v in mapping.items()}
        # embedding mapping
        mapping['Embedding-Mapping'] = [
            'electra/embeddings_project/kernel',
            'electra/embeddings_project/bias',
        ]
        return mapping

    def apply_task_related(self, inputs):
        return inputs


class NEZHA(BERT):
    """
    出自华为诺亚方舟实验室的NEZHA模型，结构上的改进是替换bert中的绝对位置编码为相对位置编码

    ref: [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](http://arxiv.org/abs/1909.00204)
    """
    def apply_embeddings(self, inputs):
        """
        embedding 是 token embedding 与 segment embedding 的和
        """
        x, s = inputs[:2]
        token_embedding = self.apply(inputs=x,
                                     layer=Embedding,
                                     name='Embedding-Token',
                                     input_dim=self.vocab_size,
                                     output_dim=self.embedding_size,
                                     embeddings_initializer=self.initializer,
                                     mask_zero=True
                                     )
        segment_embedding = self.apply(s,
                                       Embedding,
                                       name='Embedding-Segment',
                                       input_dim=2,
                                       output_dim=self.embedding_size,
                                       embeddings_initializer=self.initializer,
                                       )
        x = self.apply([token_embedding, segment_embedding], Add, name='Embedding-Token-Segment')

        x = self.apply(x,
                       LayerNormalization,
                       name='Embedding-Norm')
        x = self.apply(x,
                       Dropout,
                       name='Embedding-Dropout',
                       rate=self.dropout_rate)
        if self.hidden_size != self.embedding_size:
            x = self.apply(x,
                           Dense,
                           name='Embedding-Mapping',
                           units=self.hidden_size,
                           kernel_initializer=self.initializer)

        return x

    def apply_transformer_layers(self, inputs, idx):
        """
        Att --> Dropout --> Add --> LN --> FFN --> Dropout -->  Add --> LN
        """
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % idx
        feed_forward_name = 'Transformer-%d-FeedForward' % idx
        attention_mask = self.compute_attention_mask(idx)
        position_bias = self.compute_position_bias(inputs)

        x_pre, x = inputs, [inputs, inputs, inputs, position_bias]  # 加入相对位置编码
        arguments = {'a_mask': None, 'position_bias': 'relative'}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.insert(3, attention_mask)

        # self-attention
        x = self.apply(x,
                       MultiHeadAttention,
                       name=attention_name,
                       head_nums=self.num_attention_heads,
                       head_size=self.attention_head_size,
                       arguments=arguments,
                       kernel_initializer=self.initializer)

        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % attention_name,
                       rate=self.dropout_rate)

        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % attention_name
                       )

        x = self.apply(x,
                       LayerNormalization,
                       name='%s-Norm' % attention_name,
                       )

        # feedforward
        x_pre = x
        x = self.apply(x,
                       FeedForward,
                       name=feed_forward_name,
                       units=self.intermediate_size,
                       activation=self.hidden_act,
                       kernel_initializer=self.initializer
                       )
        x = self.apply(x,
                       Dropout,
                       name='%s-Dropout' % feed_forward_name,
                       rate=self.dropout_rate)
        x = self.apply([x_pre, x],
                       Add,
                       name='%s-Add' % feed_forward_name)
        x = self.apply(x, LayerNormalization, name='%s-Norm' % feed_forward_name)
        return x

    def compute_position_bias(self, inputs):
        """计算相对位置编码"""
        if self.position_bias is None:

            x = inputs
            self.position_bias = self.apply(
                inputs=[x, x],
                layer=RelativePositionEmbedding,
                name='Relative-Position-Embedding',
                input_dim=2 * 64 + 1,
                output_dim=self.attention_key_size,
                embedding_initializer='sinusoidal',
                trainable=False
            )

        return self.position_bias


class DBERT(BERT):
    """
    用连续的dgcnn来替换原始bert中的transformer block，来尝试压缩bert
    """

    def __init__(self, inner_hidden_size=None, **kwargs):
        prefix = kwargs.get('prefix', 'DBERT-')
        kwargs['prefix'] = prefix
        super(DBERT, self).__init__(**kwargs)
        self.inner_hidden_size = inner_hidden_size or self.attention_head_size  # dgcnn hidden dim

    def apply_transformer_layers(self, inputs, idx):
        """
            liner_in(inner_hid_dim) -> dgcnn(1) -> dgcnn(3) -> dgcnn(5) -> dgcnn(8) -> liner_out(hid_dim)
        """
        block_name = 'DBlock-%d-MultiDGCNN' % idx
        liner_in_name = 'DBlock-%d-MultiDGCNN-LinerIn' % idx
        liner_out_name = 'DBlock-%d-MultiDGCNN-LinerOut' % idx
        attention_mask = self.compute_attention_mask(idx)
        x = inputs
        x = self.apply(x,
                       Dense,
                       name=liner_in_name,
                       units=self.inner_hidden_size,
                       use_bias=False
                       )
        arguments = {}
        if attention_mask is not None:
            x.append(attention_mask)

        # dgcnn-block
        x = self.apply(x,
                       DGCNN,
                       name='{block_name}-DilationRate-{r}'.format(block_name=block_name, r=1),
                       dilation_rate=1,
                       dropout_rate=self.dropout_rate,
                       arguments=arguments)

        x = self.apply(x,
                       DGCNN,
                       name='{block_name}-DilationRate-{r}'.format(block_name=block_name, r=3),
                       dilation_rate=3,
                       dropout_rate=self.dropout_rate,
                       arguments=arguments)
        x = self.apply(x,
                       DGCNN,
                       name='{block_name}-DilationRate-{r}'.format(block_name=block_name, r=5),
                       dilation_rate=5,
                       dropout_rate=self.dropout_rate,
                       arguments=arguments)
        x = self.apply(x,
                       DGCNN,
                       name='{block_name}-DilationRate-{r}'.format(block_name=block_name, r=8),
                       dilation_rate=8,
                       dropout_rate=self.dropout_rate,
                       arguments=arguments)

        x = self.apply(x,
                       Dense,
                       name=liner_out_name,
                       units=self.hidden_size,
                       use_bias=False
                       )
        x = self.apply(x, LayerNormalization, name='DBlock-%d-MultiDGCNN-Norm' % idx)
        return x


def extend_with_language_model(BaseModel):
    """
    增加下三角mask矩阵，作为语言模型使用
    """

    class LanguageModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True  # mlm output

        def compute_attention_mask(self, inputs=None):
            """重写attention mask 计算逻辑：全局下三角矩阵，形如：
            [[[[1, 0, 0]
            [1, 1, 0]
            [1, 1, 1]]]]
            """
            if self.attention_mask is None:
                def compute_lm_mask(segments):
                    seq_len = K.shape(segments)[1]
                    idx = K.arange(0, seq_len)
                    mask = idx[:, None] <= idx[None, :]
                    mask = K.cast(mask, K.floatx())
                    return mask[None, None]

                self.attention_mask = self.apply(inputs=self.inputs[1],
                                                 layer=Lambda,
                                                 function=compute_lm_mask,
                                                 name='Attention-LM-Mask')

            return self.attention_mask

    return LanguageModel


def extend_with_unilm(BaseModel):
    """添加UniLM mask"""

    class UniLM(BaseModel):
        """UnilM-V1: https://arxiv.org/pdf/1905.03197.pdf"""

        def __init__(self, *args, **kwargs):
            super(UniLM, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True  # mlm output

        def compute_attention_mask(self, inputs=None):
            """重写attention mask 计算逻辑,segment 1 全部为1，segment 2 中为下三角矩阵.
            假如 输入序列为 [CLS] [seg1] [SEP] [seg2] [SEP], 对应 mask 为：
            [[1，1，1, 0, 0]
            [1, 1, 1, 0, 0]
            [1, 1, 1, 0, 0]
            [1, 1, 1, 1, 0]
            [1, 1, 1, 1, 1,]]
            """
            if self.attention_mask is None:
                def compute_unilm_mask(segments):
                    idx = K.cumsum(segments, axis=1)
                    mask = idx[:, None, :] <= idx[:, :, None]
                    mask = K.cast(mask, K.floatx())
                    return mask[:, None]

                self.attention_mask = self.apply(inputs=self.inputs[1],
                                                 layer=Lambda,
                                                 function=compute_unilm_mask,
                                                 name='Attention-UniLM-Attention')

            return self.attention_mask

    return UniLM


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
    # convert config name
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')

    models = {
        'bert': BERT,
        'reberta': BERT,
        'nezha': NEZHA,
        'electra': ELECTRA,
        'dbert': DBERT
    }

    if isinstance(model, six.string_types):
        model = model.lower()
        MODEL = models[model]
    else:
        MODEL = model

    application = application.lower()
    if application == 'lm':
        MODEL = extend_with_language_model(MODEL)
    elif application == 'unilm':
        MODEL = extend_with_unilm(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model
    else:
        return transformer
