# -*- coding: utf-8 -*-
# @Date    : 2020/8/6
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : sequence_labeling_cws_crf.py
"""
CRF做中文分词（Chinese Word Segment）
数据集 http://sighan.cs.uchicago.edu/bakeoff2005/
官方评测F1：96%
"""

import re
from tqdm import tqdm
import os
import numpy as np

from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.layers import Dense
from toolkit4nlp.utils import ViterbiDecoder
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.optimizers import Adam
from toolkit4nlp.backend import keras, K
from toolkit4nlp.utils import DataGenerator
from toolkit4nlp.utils import pad_sequences
from toolkit4nlp.layers import ConditionalRandomField
from toolkit4nlp.optimizers import extend_with_gradient_accumulation

train_path = '/home/mingming.xu/datasets/NLP/segment/icwb-2-data/training/pku_training.utf8'
test_path = '/home/mingming.xu/datasets/NLP/segment/icwb-2-data/testing/pku_test.utf8'
data_dir = '/home/mingming.xu/datasets/NLP/segment/icwb-2-data/'
test_result_path = 'test_result.txt'
test_score_path = 'test_score.txt'

bert_config = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
bert_dict = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'
bert_checkpoint = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'

maxlen = 256
bert_layers = 12
lr_multi = 2  # crf loss 放大
num_labes = 4
lr = 1e-5
batch_size = 16  # 实际梯度在此基础上累加2次，即 batch_size = 32
epochs = 5

tokenizer = Tokenizer(bert_dict, do_lower_case=True)


def load_data(data_path):
    items = []
    with open(data_path) as f:
        for line in f:
            chunk = re.split(' +', line.strip())
            items.append(chunk)
    return items


data = load_data(train_path)

row_nums = list(range(len(data)))
np.random.shuffle(row_nums)

train_data = [data[i] for i in row_nums if i % 10 != 0]
val_data = [data[i] for i in row_nums if i % 10 == 0]
print(len(train_data), len(val_data))

class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        # 0: 单字，1，多字词开头，2，中间，3，末尾
        batch_tokens, batch_segs, batch_labels = [], [], []
        for is_end, item in self.get_sample(shuffle):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for word in item:
                token_id = tokenizer.encode(word)[0][1:-1]
                if len(token_ids) + len(token_id) > maxlen:
                    break
                if len(token_id) == 1:
                    labels += [0]
                else:
                    labels += [1] + [2] * (len(token_id) - 2) + [3]
                token_ids += token_id

            token_ids.append(tokenizer._token_end_id)
            labels.append(0)
            batch_tokens.append(token_ids)
            batch_segs.append([0] * len(token_ids))
            batch_labels.append(labels)

            if len(batch_tokens) >= self.batch_size or is_end:
                batch_tokens = pad_sequences(batch_tokens)
                batch_segs = pad_sequences(batch_segs)
                batch_labels = pad_sequences(batch_labels)
                yield [batch_tokens, batch_segs], batch_labels
                batch_tokens, batch_segs, batch_labels = [], [], []


model = build_transformer_model(config_path=bert_config, checkpoint_path=bert_checkpoint)
output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labes)(output)
CRF = ConditionalRandomField(lr_multi)
output = CRF(output)
model = Model(model.input, output)
model.summary()


class WordSeg(ViterbiDecoder):
    def segment(self, data):
        tokens = tokenizer.tokenize(data)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(data, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segs = [0] * len(token_ids)
        pre = model.predict([[token_ids], [segs]])[0]
        labels = self.decode(pre)

        words = []
        for i, label in enumerate(labels[1:-1]):
            if label < 2 or len(words)==0:
                words.append([i + 1])
            else:
                words[-1].append(i + 1)
        return [data[mapping[w[0]][0]: mapping[w[-1]][-1] + 1] for w in words]


wordseg = WordSeg(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """简单评测"""
    total, right = 1e-10, 1e-10
    for true in tqdm(data):
        pre = wordseg.segment(''.join(true))
        w_pre = set(pre)
        w_true = set(true)
        total += len(w_true)
        right += len(w_pre & w_true)

    return right / total


def public_evaluate(test_path, test_result_path, test_score_path):
    """官方评测，结果在score file 的最后几行"""
    fw = open(test_result_path, 'w', encoding='utf-8')
    with open(test_path, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = l.strip()
            if l:
                l = ' '.join(wordseg.segment(l))
            fw.write(l + '\n')
    fw.close()
    # 运行官方评测脚本
    with os.popen(
        '{data_dir}/scripts/score {data_dir}/gold/pku_training_words.utf8 {data_dir}/gold/pku_test_gold.utf8 {test_result_path} > {test_score_path}'.format(
            data_dir=data_dir, test_result_path=test_result_path, test_score_path=test_score_path)) as p:
        p.read()

    # 打印结果的最后两行
    result = open(test_score_path).readlines()
    for line in result[-2:]:
        print(line)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        wordseg.trans = trans
        print(trans)
        acc = evaluate(val_data)

        if acc > self.best_acc:
            self.best_acc = acc
            model.save_weights('./best_model.weights')
        print('acc is: {:.3f}, best acc is :{:.4f}'.format(acc, self.best_acc))

    def on_train_end(self, logs=None):
        model.load_weights('./best_model.weights')
        public_evaluate(test_path, test_result_path, test_score_path)


opt = extend_with_gradient_accumulation(Adam)
opt = opt(learning_rate=lr)
model.compile(loss=CRF.sparse_loss, optimizer=opt, metrics=[CRF.sparse_accuracy])

if __name__ == '__main__':
    evaluator = Evaluator()
    train_genarator = data_generator(train_data, batch_size)
    model.fit_generator(
        train_genarator.generator(),
        steps_per_epoch=len(train_genarator),
        epochs=epochs,
        callbacks=[evaluator])
else:
    model.load_weights('./best_model.weights')
