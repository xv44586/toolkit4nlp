# -*- coding: utf-8 -*-
# @Date    : 2020/9/18
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : classification_ifytek_fastbert.py
"""
借鉴fastbert 的思想：对不同样本选择不同的Transformer 层进行预测，来达到提前结束计算，加速推理。
实验时发现，由于没有大量unlabel data，只利用Teacher model outputs进行迁移，效果非常差，所以选择迁移时对每个
branch classifier 也迁移ground truth，同时通过简单的句子重复与打散来进行数据增强。

**注意**：由于实验中使用的K.switch 并不是lazier semantics,所以并不能真正达到跳过计算，而由于增加了分支结果的判断，
所以实际上比Teacher model推理更慢，如果有更好的方式，请issue

ref: [FastBERT](http://arxiv.org/abs/2004.02178)
"""


import json
from tqdm import tqdm
import re

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
num_hidden_layers = 6
speed = 0.1  # uncertainty阈值
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
    """迁移时由于没有额外的label data，所以通过data augmentation 来模拟。
    方法是切分句子后重复后shuffle再重新组成新的句子
    """

    def __init__(self, data_augmentation=False, transfer=False, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation
        self.transfer = transfer

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.get_sample():
            if self.data_augmentation:
                text = self.generate_text(text)
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                if not self.transfer:
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                else:
                    yield [batch_token_ids, batch_segment_ids] + [batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def generate_text(self, text):
        pat = '[,.?!，。？！；;]+'
        sentences = re.split(pat, text)
        sentences = sentences * 2
        np.random.shuffle(sentences)
        return '。'.join(sentences)


# 转换数据集
train_generator = data_generator(data=train_data, batch_size=batch_size)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)
train_transfer_generator = data_generator(data=train_data,
                                          batch_size=batch_size, transfer=True, data_augmentation=True)

# 加载预训练模型（3层）
teacher = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    num_hidden_layers=num_hidden_layers,
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
    """FastBert 中用来做分类的层，为了增加分类层的性能，同时参数不能太大，所以作者选择了一个hidden size
    更小的transformer
    """

    def __init__(self, labels_num, hidden_size=128, head_nums=2, head_size=64, pooling=None, **kwargs):
        super(FastbertClassifierLayer, self).__init__(**kwargs)
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
        return input_shape[:1] + (self.labels_num,)


def normal_shannon_entropy(p, labels_num=num_classes):
    # normalized entropy
    p = K.cast(p, K.floatx())
    norm = K.log(1. / labels_num)
    s = K.sum(p * K.log(p), axis=-1, keepdims=True)
    return s / norm


class SwitchTwo(Layer):
    """通过classifier 的结果，来选择是否跳过下一层计算
    **注意**：由于tf.cond 对function中含有任意tensor/operation 时，两个function都会执行，
    所以这里并没有达到 “跳过”的逻辑，暂时没找到更好的方式来实现。
    关于tf.cond，请参考：
    [control_flow_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/control_flow_ops.py#L1105)

    """

    def __init__(self, speed=0.1, *args, **kwargs):
        super(SwitchTwo, self).__init__(*args, **kwargs)
        self.supports_masking = True
        self.speed = K.constant(speed, dtype=float)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[-1]

    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    def call(self, inputs):
        clf, x_pre, x_next = inputs
        uncertain = normal_shannon_entropy(clf, num_classes)
        cond = K.greater(self.speed, uncertain)
        x = K.switch(cond, x_pre, x_next)
        return K.in_train_phase(x_next, x)


def fastbert(teacher, classifier, speed=speed):
    inputs = teacher.inputs
    # frozen layers
    for layer in teacher.model.layers:
        layer.trainable = False
    classifier.trainable = False

    x_pre = teacher.apply_embeddings(inputs)
    emb_name = 'FastBert-embedding'
    clf_pre = teacher.apply(x_pre, FastbertClassifierLayer, name=emb_name, labels_num=num_classes)
    student_outputs = [clf_pre]
    outputs = [clf_pre, x_pre]

    for idx in range(teacher.num_hidden_layers):
        clf_pre, x_pre = outputs
        name = 'FastBert-%d' % idx
        x_next = teacher.apply_transformer_layers(x_pre, idx)
        clf_next = teacher.apply(x_pre, FastbertClassifierLayer, name=name, labels_num=num_classes)
        student_outputs.append(clf_next)

        x = SwitchTwo(speed)([clf_pre, x_pre, x_next])
        clf = SwitchTwo(speed)([clf_pre, clf_pre, clf_next])
        outputs = [clf, x]

    clf_prob, x = outputs
    x = classifier(x)

    output = SwitchTwo(speed)([clf_prob, clf_prob, x])
    model_infer = Model(inputs, output)

    label_inputs = Input(shape=(None,))
    model_train = Model(inputs + [label_inputs], student_outputs)

    for i, prob in enumerate(student_outputs):
        ce_loss = K.sparse_categorical_crossentropy(label_inputs, prob)
        kl_loss = kullback_leibler_divergence(x, prob)
        model_train.add_loss(ce_loss)
        model_train.add_metric(ce_loss, name='ce_loss-%d' % i)
        model_train.add_loss(kl_loss)
        model_train.add_metric(kl_loss, name='loss-%d' % i)

    model_1 = Model(inputs, student_outputs[1])
    model_2 = Model(inputs, student_outputs[2])

    return model_train, model_infer, model_1, model_2


model_train, model_infer, model_1, model_2 = fastbert(teacher, classifier, speed=speed)

model_train.compile(optimizer=Adam(1e-5))


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


def evaluate_single(data, model):
    """验证单个样本"""
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        for i in range(len(y_true)):
            tokens, segs = x_true
            token = tokens[i: i + 1]
            seg = segs[i: i + 1]
            x = [token, seg]
            y = y_true[i]
            y_pred = model.predict(x).argmax(axis=1)
            total += len(y)
            right += (y == y_pred).sum()
    print(right, total)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_name, evaluate_model=None, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        self.save_name = save_name
        self.evaluate_model = evaluate_model
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        if self.evaluate_model is None:
            self.evaluate_model = self.model

        cur_acc = evaluate(valid_generator, self.evaluate_model)
        if self.best_acc < cur_acc:
            self.best_acc = cur_acc
            self.model.save(self.save_name)
        print('cur acc: ', cur_acc, ' best acc: ', self.best_acc)


print(evaluate(valid_generator, model_1))
print(evaluate(valid_generator, model_2))

if __name__ == '__main__':
    teacher_model_name = 'best.teacher.weights'
    teacher_evaluator = Evaluator(teacher_model_name)
    teacher_model.fit_generator(train_generator.generator(),
                                steps_per_epoch=len(train_generator),
                                epochs=5,
                                callbacks=[teacher_evaluator])

    # train fastbert
    fastbert_model_name = 'best.fastbert.weights'
    fastbert_evaluator = Evaluator(fastbert_model_name, model_1)
    model_train.fit_generator(train_transfer_generator.generator(),
                              steps_per_epoch=len(train_transfer_generator),
                              epochs=20,
                              callbacks=[fastbert_evaluator])

    # evaluate single sample
    print(evaluate_single(valid_generator, model_infer))

else:
    model_name = 'best.fastbert.weights'
    model_train.load_weights(model_name)
    print(evaluate_single(valid_generator, model_infer))
