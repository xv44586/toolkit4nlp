# -*- coding: utf-8 -*-
# @Date    : 2020/9/18
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : classification_ifytek_fastbert.py
import json
from tqdm import tqdm

import numpy as np
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.optimizers import Adam, extend_with_piecewise_linear_lr
from toolkit4nlp.utils import DataGenerator, pad_sequences
from toolkit4nlp.layers import *
from keras.models import Model
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.losses import kullback_leibler_divergence

num_classes = 119
maxlen = 128
batch_size = 32
max_label = 7
# BERT base
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], l['label_des']
            D.append((text, int(label), label_des))
    return D


# 加载数据集
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/train.json'
)
valid_data = load_data(
    '/home/mingming.xu/datasets/NLP/CLUE/iflytek/dev.json'
)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    vocab_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, has_label=True, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        self.has_label = has_label

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.get_sample():
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                if self.has_label:
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                else:
                    yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(data=train_data, batch_size=batch_size)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)
train_no_label_generator = data_generator(data=train_data, batch_size=batch_size, has_label=False)


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true)[:, :num_classes].argmax(axis=1)
        if y_true.shape[1] > 1:
            y_true = y_true[:, :num_classes].argmax(axis=-1)
        else:
            y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


# 加载预训练模型（3层）
teacher = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=3,
    model='bert'
)

# 判别模型
x_in = Input(shape=K.int_shape(teacher.output)[1:])
x = Lambda(lambda x: x[:, 0])(x_in)
x = Dense(units=num_classes, activation='softmax')(x)
classifier = Model(x_in, x)

teacher_model = Model(teacher.inputs, classifier(teacher.output))
teacher_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)

teacher_model.summary()


class FastbertClassifierLayer(Layer):
    def __init__(self, labels_num, hidden_size=128, head_nums=2, head_size=64, pooling=None, **kwargs):
        super(FastbertClassifierLayer, self).__init__(**kwargs)
        #         self.idx = idx
        self.labels_num = labels_num
        self.hidden_size = hidden_size
        self.head_nums = head_nums
        self.head_size = head_size
        self.pooling = pooling

    def build(self, input_shape):
        #         prefix = 'FastBert-%d-Classifier' % self.idx
        prefix = self.name
        self.fc_1 = Dense(units=self.hidden_size, name=prefix + '-fc-1')
        self.mul_attention = MultiHeadAttention(head_nums=self.head_nums,
                                                head_size=self.head_size,
                                                name=prefix + '-MultiHead')
        self.fc_2 = Dense(units=self.hidden_size, name=prefix + '-fc-2')
        self.fc_3 = Dense(units=self.labels_num, activation='softmax', name=prefix + '-fc-3')

    def call(self, inputs, mask=None):
        """
        FC(128) -> self-attention(128) -> pooling() -> FC(128) -> FC(n)
        pooling: 默认取 [cls] 对应hidden state, 可选 mean/max/last
        """
        x = self.fc_1(inputs)
        x = [x, x, x]
        x = self.mul_attention(x)
        # pooling
        x = Lambda(lambda x: x[:, 0])(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:] + (self.labels_num,)


def normal_shannon_entropy(p, labels_num):
    # normalized entropy
    p = K.cast(p, K.floatx())
    norm = K.log(1. / labels_num)
    s = K.sum(p * K.log(p))
    return s / norm


def fastbert(teacher, classifier, speed=0.1):
    inputs = teacher.inputs
    # frozen layers
    for layer in teacher.model.layers:
        layer.trainable = False
    classifier.trainable = False

    x = teacher.apply_embeddings(inputs)
    first_name = 'FastBert-0'
    clf_prob = teacher.apply(x, FastbertClassifierLayer, name=first_name, labels_num=num_classes)
    x = (clf_prob, x)
    student_outputs = [clf_prob]

    for idx in range(teacher.num_hidden_layers):
        clf_prob, x = x
        name = 'FastBert-%d' % idx
        x = teacher.apply_attention_layers(x, idx)
        clf_prob = teacher.apply(x, FastbertClassifierLayer, name=name, labels_num=num_classes)
        student_outputs.append(clf_prob)
        x = [clf_prob, x]

    clf_prob, x = x
    x = classifier(x)
    model = Model(inputs, [x] + student_outputs)
    model_infer = Model(inputs, x)
    model_1 = Model(inputs, student_outputs[0])
    model_2 = Model(inputs, student_outputs[1])
    model_3 = Model(inputs, student_outputs[2])
    for prob in student_outputs:
        model.add_loss(kullback_leibler_divergence(x, prob))
    return model, model_infer, model_1, model_2, model_3


model, model_infer, model_1, model_2, model_3 = fastbert(teacher, classifier, speed=0.1)
print(evaluate(valid_generator, model_1))
print(evaluate(valid_generator, model_2))
print(evaluate(valid_generator, model_3))

Adamw = extend_with_piecewise_linear_lr(Adam)
opt = Adamw(lr_schedule={3 * len(train_no_label_generator): 1., 10 * len(train_no_label_generator): 0.},
            learning_rate=2e-5)
model.compile(optimizer=opt, metrics=['sparse_categorical_accuracy'])

if __name__ == '__main__':
    teacher_model.fit_generator(train_generator.generator(),
                                steps_per_epoch=len(train_generator),
                                epochs=5)

    # train fastbert

    model.fit_generator(train_no_label_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=10)

    # infer
    # todo