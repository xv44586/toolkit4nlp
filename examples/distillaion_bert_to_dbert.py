# -*- coding: utf-8 -*-
# @Date    : 2020/9/3
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : distillaion_bert_to_dbert.py
"""
用膨胀系数为1/3/5/8 四种cnn组成的block来蒸馏bert中的transformer，loss是 embedding_output + transformer_output + classifier_output + label_loss
其中除了label_loss外，对各个层输出的蒸馏都是采用 MSE + CrossEntropy，其中MSE用来逼近数值，CE用来逼近分布。
任务是ner，主体结构是bert + softmax
"""
import os
from tqdm import tqdm
from toolkit4nlp.backend import K, keras
from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.utils import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.optimizers import *

data_dir = '/home/mingming.xu/datasets/NLP/ner/china-people-daily-ner-corpus/'
train_path = os.path.join(data_dir, 'example.train')
test_path = os.path.join(data_dir, 'example.test')
val_path = os.path.join(data_dir, 'example.dev')

bert_layers = 12
maxlen = 256
batch_size = 24
crf_lr_multiplier = 1
learning_rate = 1e-5
epochs = 5


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


train_data = load_data(train_path)
test_data = load_data(test_path)
valid_data = load_data(val_path)

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 类别映射
labels = ['PER', 'LOC', 'ORG']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 3 + 1


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.get_sample():
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 3 + 1
                        I = label2id[l] * 3 + 2
                        E = label2id[l] * 3 + 3

                        labels += ([B] + [I] * (len(w_token_ids) - 2) + [E] * int(len(w_token_ids) > 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                #                 softmax need label has shape: (batch_size, sequence_length, 1)
                batch_labels = np.expand_dims(batch_labels, -1)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def decode(self, nodes):
        #         nodes = nodes[1:-1]
        return np.argmax(nodes, axis=-1)

    def recognize(self, model, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids = np.array([token_ids])
        segment_ids = np.array([segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels[1:-1]):
            if label > 0:
                if label % 3 == 1:
                    starting = True
                    entities.append([[i + 1], id2label[(label - 1) // 3]])
                elif starting:
                    entities[-1][0].append(i + 1)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


# NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
NER = NamedEntityRecognizer(trans=np.zeros((num_labels, num_labels)), starts=[0], ends=[0])


def evaluate(model, data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(model, text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self, model_name):
        self.best_val_f1 = 0
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        #         trans = K.eval(CRF.trans)
        #         NER.trans = trans
        #         print(NER.trans)
        f1, precision, recall = evaluate(self.model, valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_name)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(self.model, test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


bert = build_transformer_model(
    config_path,
    checkpoint_path,
    return_keras_model=False
)

x_in = Input(shape=K.int_shape(bert.output)[1:])
x = Lambda(lambda x: x)(x_in)
# softmax
x = Dense(num_labels, activation='softmax')(x)
bert_classifier = Model(x_in, x)

teacher_model = Model(bert.input, bert_classifier(bert.output))
teacher_model.summary()

teacher_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['sparse_categorical_accuracy']
)


if __name__ == '__main__':
    teacher_model_name = './best_teacher_model.weights'
    teacher_evaluator = Evaluator(teacher_model_name)
    train_generator = data_generator(train_data, batch_size)

    teacher_model.fit(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[teacher_evaluator]
    )