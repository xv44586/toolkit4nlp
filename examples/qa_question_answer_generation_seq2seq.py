#!/usr/bin/env python
# -*- coding: utf8 -*-
# @Date     : 2020/08/24
# @Author   : mingming.xu
# @Email    : xv44586@gmail.com
"""
利用 bert + unilm 生成问题+回答
"""
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm

from toolkit4nlp.backend import K, keras
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import Model, build_transformer_model
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.utils import DataGenerator, pad_sequences, text_segmentate, AutoRegressiveDecoder

# 基本信息
max_context_len = 256
max_question_len = 64
max_answer_len = 16
epochs = 20
batch_size = 16
learning_rate = 1e-5

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for context_seg in text_segmentate(d['context'], max_context_len):
            for qa in d['qas']:
                for answer in [a['text'] for a in qa.get('answers', [])]:
                    if answer not in context_seg:
                        continue

                    D.append([
                        qa['id'], context_seg, qa['question'], answer
                    ])
    return D


# 读取数据
train_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/train.json'
)
val_data = load_data(
    '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/dev.json'
)
# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    vocab_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self):
        """[CLS]context[SEP]answer[SEP]question[SEP]"""
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
            context, question, answer = item[1:]
            c_token_ids, _ = tokenizer.encode(context, maxlen=max_context_len + 1)
            q_token_ids, _ = tokenizer.encode(question, maxlen=max_question_len)
            a_token_ids, _ = tokenizer.encode(answer, maxlen=max_answer_len)

            token_ids = c_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * len(c_token_ids) + [1] * (len(token_ids) - len(c_token_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


# loss 层，错位计算预测值并mask掉segment1
class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# build model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class QuestionAnswerGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps('probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        ret = model.predict([token_ids, segment_ids])[:, -1]
        return ret

    def generate(self, context, topk=5):
        """随机生成答案，再用beam search生成对应问题"""
        token_ids, segment_ids = tokenizer.encode(context, maxlen=max_context_len)
        segment_ids = [0] * len(token_ids)
        # 随机生成答案
        answer_ids = self.random_sample([token_ids, segment_ids], 1, topk)[0]

        token_ids += list(answer_ids)
        segment_ids += [1] * len(answer_ids)
        # 随机解码，用于生成新的question
        question_ids = self.beam_search(inputs=[token_ids, segment_ids], beam_size=topk)
        return tokenizer.decode(answer_ids), tokenizer.decode(question_ids)


question_answer_generator = QuestionAnswerGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=max_question_len)


def generate_question_answer(context):
    return question_answer_generator.generate(context)


def just_show():
    idx = np.random.choice(len(train_data), 3)
    for i in idx:
        sample = train_data[i]
        print(u'context：%s' % sample[1])
        print(u'question：%s ' % sample[2])
        print(u'answer: %s' % sample[3])
        new_answer, new_question = generate_question_answer(sample[1])
        print('generate question: %s ' % new_question)
        print('generate answer: %s ' % new_answer)


class Evaluator(keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.lowest_loss = 1e4

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['loss']
        if current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.model.save_weights('question_answer_generation.weights')
        print('epoch: {},loss: {}, lowest loss: {}'.format(epoch, current_loss, self.lowest_loss))
        just_show()


def generate_new_data(file_name='train_qa_generation.json'):
    paras = defaultdict(list)

    for data in tqdm(train_data):
        id_, context, question, answers = data
        paras[context].append({'id': id_, 'question': question, 'answers': [{'text': answers}]})
        new_question, new_answer = generate_question_answer(context)
        if new_question != question:
            paras[context].append({'id': id_, 'question': new_question, 'answers': [{'text': new_answer}]})
    paragraphs = []
    for context, qas in paras.items():
        paragraphs.append({'context': context, 'qas': qas})

    data = {'data': [{'paragraphs': paragraphs}]}

    with open(file_name, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    train_generator = data_generator(train_data + val_data, batch_size)
    evaluator = Evaluator()
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

    # generate question and answer
    model.load_weights('question_answer_generation.weights')
    file_name = '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/train_qa_generator.json'
    generate_new_data(file_name)

else:
    model.load_weights('question_answer_generation.weights')
    file_name = '/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/train_qa_generator.json'
    generate_new_data(file_name)