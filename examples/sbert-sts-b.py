# -*- coding: utf-8 -*-
# @Date    : 2021/1/13
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : sbert-sts-b.py
"""
data:
  [STSbenchmark](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)
  [snli](https://nlp.stanford.edu/projects/snli/)
paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](http://arxiv.org/abs/1908.10084)
"""
import os
import json
from tqdm import tqdm
from scipy.stats import spearmanr

from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.models import *
from toolkit4nlp.optimizers import *

from toolkit4nlp.tokenizers import *
from toolkit4nlp.utils import *

label2id = {'neutral': 0, 'entailment': 1, 'contradiction': 2}


def load_snli_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            label = item['gold_label']
            s1 = item['sentence1']
            s2 = item['sentence2']
            if label not in label2id:
                continue

            label_id = label2id[label]
            D.append([s1, s2, label_id])
    return D


snli_train = load_snli_data('/home/mingming.xu/datasets/NLP/GLUE/snli_1.0/snli_1.0_train.jsonl')
snli_test = load_snli_data('/home/mingming.xu/datasets/NLP/GLUE/snli_1.0/snli_1.0_test.jsonl')
snli_dev = load_snli_data('/home/mingming.xu/datasets/NLP/GLUE/snli_1.0/snli_1.0_dev.jsonl')


def load_stsb_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            if i > 0:
                l = l.strip().split('\t')
                D.append((l[5], l[6], float(l[4])))
    return D


stsb_train = load_stsb_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-train.csv')
stsb_test = load_stsb_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-test.csv')
stsb_dev = load_stsb_data('/home/mingming.xu/datasets/NLP/GLUE/STS-B/sts-dev.csv')

config_path = '/home/mingming.xu/pretrain/NLP/google_uncased_english_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/google_uncased_english_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/mingming.xu/pretrain/NLP/google_uncased_english_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)

maxlen = 128
batch_size = 16
epochs = 1
lr = 2e-5


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        token_ids_1, segment_ids_1, token_ids_2, segment_ids_2, labels = [], [], [], [], []
        for is_end, item in self.get_sample(shuffle):
            sen1, sen2, label = item

            tokens_1, segments_1 = tokenizer.encode(sen1, maxlen=maxlen)
            tokens_2, segments_2 = tokenizer.encode(sen2, maxlen=maxlen)

            token_ids_1.append(tokens_1)
            segment_ids_1.append(segments_1)
            token_ids_2.append(tokens_2)
            segment_ids_2.append(segments_2)
            labels.append([label])

            if is_end or len(token_ids_1) == self.batch_size:
                token_ids_1 = pad_sequences(token_ids_1, maxlen=maxlen)
                segment_ids_1 = pad_sequences(segment_ids_1, maxlen=maxlen)
                token_ids_2 = pad_sequences(token_ids_2, maxlen=maxlen)
                segment_ids_2 = pad_sequences(segment_ids_2, maxlen=maxlen)
                labels = pad_sequences(labels)

                yield [token_ids_1, segment_ids_1, token_ids_2, segment_ids_2], labels
                token_ids_1, segment_ids_1, token_ids_2, segment_ids_2, labels = [], [], [], [], []


train_generator = data_generator(snli_train, batch_size)
valid_generator = data_generator(stsb_train, batch_size)


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化,当前是MEAN
    """

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, prefix='Sen-', name='bert')

token_inputs_1 = Input(shape=(None,), name='x1')
segment_inputs_1 = Input(shape=(None,), name='s1')
token_inputs_2 = Input(shape=(None,), name='x2')
segment_inputs_2 = Input(shape=(None,), name='s2')

output_1 = bert([token_inputs_1, segment_inputs_1])
output_2 = bert([token_inputs_2, segment_inputs_2])

u = GlobalAveragePooling1D(name='pool_1')(inputs=output_1)
v = GlobalAveragePooling1D(name='pool_2')(inputs=output_2)

# u = Lambda(lambda x: x[:,0])(output_1)
# v = Lambda(lambda x: x[:,0])(output_2)
u_v = Lambda(lambda x: x[0] - x[1])([u, v])

x = Concatenate()([u, v, u_v])
x = Dense(3, activation='softmax')(x)

model = Model([token_inputs_1, segment_inputs_1, token_inputs_2, segment_inputs_2], x)

infer_model = Model([token_inputs_1, segment_inputs_1], u, name='encoder')

model.summary()


# infer_model.summary()

def get_sentence_vector(sentences):
    token_ids, segment_ids = [], []
    for sent in sentences:
        tokens, segments = tokenizer.encode(sent, maxlen=maxlen)
        token_ids.append(tokens)
        segment_ids.append(segments)

    token_ids = pad_sequences(token_ids)
    segment_ids = pad_sequences(segment_ids)

    vec = infer_model.predict([token_ids, segment_ids], verbose=True)
    return vec


def cal_sim(data):
    # cal cosine sim
    sentences_1 = [s[0] for s in data]
    sentences_2 = [s[1] for s in data]
    vecs_1 = get_sentence_vector(sentences_1)
    vecs_2 = get_sentence_vector(sentences_2)
    vecs_1 = vecs_1 / (vecs_1 ** 2).sum(axis=1, keepdims=True) ** 0.5
    vecs_2 = vecs_2 / (vecs_2 ** 2).sum(axis=1, keepdims=True) ** 0.5
    sims = (vecs_1 * vecs_2).sum(axis=1)
    return sims


def evaluate(data):
    # 计算相关系数
    sims = cal_sim(data)
    labels = [d[-1] for d in data]
    cor = np.corrcoef(sims, labels)[0, 1]  # Pearson correlation
    spear, _ = spearmanr(sims, labels)  # Spearman rank correlation
    return cor, spear


Opt = extend_with_weight_decay(Adam)
exclude_from_weight_decay = ['Norm', 'bias']
Opt = extend_with_piecewise_linear_lr(Opt)
para = {
    'learning_rate': 2e-5,
    'weight_decay_rate': 0.1,
    'exclude_from_weight_decay': exclude_from_weight_decay,
    'lr_schedule': {int(len(train_generator) * 0.1 * epochs): 1, int(len(train_generator) * epochs): 0},
}

opt = Opt(**para)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])


class Evaluator(keras.callbacks.Callback):
    def __init__(self, score_type='pearson', eval_steps=1000, save_path='best.weights'):
        self.score_type = score_type
        self.save_path = save_path
        self.best_score = 0.
        self.eval_steps = eval_steps

    def on_train_batch_end(self, batches, logs=None):
        if (batches + 1) % self.eval_steps == 0:
            p, s = evaluate(stsb_dev)
            if self.score_type == 'pearson':
                score = p
            else:
                score = s
            if score > self.best_score:
                self.best_score = score
                model.save_weights(self.save_path)
            print('steps is: {}, score is:{}, best score is: {}'.format(batches + 1, score, self.best_score))


if __name__ == '__main__':
    p, r = evaluate(stsb_train)
    print('before training, Pearson correlation : {},spearman rank correlation: {}'.format(p, r))
    save_path = 'best.weights'
    evaluator = Evaluator(save_path=save_path)
    model.fit_generator(train_generator.generator(),
                        epochs=epochs,
                        steps_per_epoch=len(train_generator),
                        )

    # load best weights
    model.load_weights(save_path)
    p, r = evaluate(stsb_train)
    print('after training, Pearson correlation : {},spearman rank correlation: {}'.format(p, r))
