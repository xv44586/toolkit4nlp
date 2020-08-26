# -*- coding: utf-8 -*-
# @Date    : 2020/8/20
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : qa_question_generator.py
"""
利用UniLM来通过篇章与答案，来生成问题，可以看做是qa 数据的增强
数据是百度2020icl比赛的机器阅读(http://lic2020.cipsc.org.cn/)
"""
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.layers import *
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.utils import pad_sequences, DataGenerator, AutoRegressiveDecoder
from toolkit4nlp.optimizers import Adam
from bert4keras.layers import Loss
from keras.callbacks import Callback

# 基本信息
maxlen = 512
epochs = 5
batch_size = 4
learning_rate = 2e-5
max_question_len = 32

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
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


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
            context, question, answers = item[1:]
            answer = np.random.choice(answers)
            token_ids, _ = tokenizer.encode(answer, context, maxlen=maxlen - max_question_len - 1)
            segment_ids = [0] * len(token_ids)

            question_token_ids = tokenizer.encode(question)[0][1:]
            token_ids = token_ids + question_token_ids
            segment_ids += [1] * len(question_token_ids)

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


class QuestionGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps('probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        ret = model.predict([token_ids, segment_ids])[:, -1]
        return ret

    def generate(self, context, answer, topk=2, random=False):
        max_q_len = maxlen - self.maxlen - 1
        token_ids, _ = tokenizer.encode(context, answer, maxlen=max_q_len)
        segment_ids = [0] * len(token_ids)
        # 确定解码，用于评估模型
        if not random:
            output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
            return tokenizer.decode(output_ids)
        # 随机解码，用于生成新的question
        output_ids = self.random_sample(inputs=[token_ids, segment_ids], n=3, topk=3, topp=0.9)
        return [tokenizer.decode(ids) for ids in output_ids]


question_generator = QuestionGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def extract_question(context, answer):
    """生成确定性问题函数
    """
    question = question_generator.generate(context, answer)
    return question


def generate_question(context, answer):
    """生成多样化问题"""
    questions = question_generator.generate(context, answer, random=True)
    return questions


def evaluate(valid_data):
    """rouge """
    T, R, P = 0., 0., 0.
    for d in tqdm(valid_data):
        _, context, question, answers = d
        answer = answers[0]
        T += len(set(question))
        r = extract_question(context, answer)
        R += len(set(r))
        P += len(set(question) & set(r))

    f1, precision, recall = 2 * P / (R + T), P / T, P / R
    return f1, precision, recall


def just_show():
    """随机观察一些样本的效果
    """
    idx = np.random.choice(len(train_data), 3)
    for i in idx:
        sample = train_data[i]
        print(u'context：%s' % sample[1])
        print(u'question：%s ' % sample[2])
        print(u'answer: %s' % sample[3])
        new_question = extract_question(sample[1], sample[3][0])
        print('generate question: %s ' % new_question)


class Evaluator(Callback):
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
        self.best_f1 = 0.

    def on_batch_end(self, batch, logs=None):
        f1, _, _ = evaluate(val_data)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights('best_model.weights')

        print('epoch {}: f1: {}, best f1: {}'.format(epochs, f1, self.best_f1))
        just_show()


if __name__ == '__main__':
    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()
    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator])

    # generate new questions
    model.load_weights('best_model.weights')
    new_train_data = []
    tem = defaultdict(list)
    for d in tqdm(train_data):
        qas = []
        _id, context, question, answers = d
        new_questions = [question]
        for answer in answers:
            g_questions = generate_question(context, answer)
            g_questions = [q for q in set(g_questions) if q not in new_questions]
            new_questions.extend(g_questions)
        answers = [{'text': a} for a in answers]
        for q in new_questions:
            qa = {'id': _id, 'question': q, 'answers': answers}
            qas.append(qa)

        tem[context].extend(qas)

    paragraphs = []
    for c, qas in tem.items():
        paragraphs.append({'context': c, 'qas': qas})

    data = {'data': [{'paragraphs': paragraphs}]}

    with open('/home/mingming.xu/datasets/NLP/qa/dureader_robust-data/new_train.json', 'w') as f:
        json.dump(data, f)

else:
    model.load_weights('best_model.weights')
