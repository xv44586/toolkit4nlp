# -*- coding: utf-8 -*-
# @Date    : 2020/7/20
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : preprocess.py
import numpy as np
import tensorflow as tf
from toolkit4nlp.backend import K


class TrainingDataset(object):
    def __init__(self, tokenizer, seq_length):
        """

        :param tokenizer: tokenizer
        :param seq_length:  sequence length
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_start_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size

    def process(self, corpus, record_name):
        """将语料（corpus）处理为tfrecord格式，并写入 record_name"""
        writer = tf.io.TFRecordWriter(record_name)
        instances = self.paragraph_process(corpus)
        instances_serialized = self.tfrecord_serialize(instances)
        for instance_serialized in instances_serialized:
            writer.write(instance_serialized)

    def get_start_end_padding_tokens(self):
        """get start/end/padding tokens for instance"""
        raise NotImplementedError

    def paragraph_process(self, corpus):
        """
        将句子不断的塞进一个instance里，直到长度即将超出seq_length
        :param corpus: sentence list构成的paragraph
        :param starts_tokens:  每个序列开始的token_id
        :param ends_tokens:   每个序列结束的token_id
        :param paddings_tokens:  每个序列padding_token
        :return: instances
        """
        starts_tokens, ends_tokens, paddings_tokens = self.get_start_end_padding_tokens()
        instances, instance = [], [starts_tokens[i] for i in starts_tokens]

        for sentence in corpus:
            sentence_tokens = self.sentence_process(sentence)
            sentence_tokens = [tokens[:self.seq_length - 2] for tokens in sentence_tokens]  # 单个句子长度不能超过最大限度
            length_after_added = len(instance[0]) + len(sentence_tokens[0])
            #  判断加入这句长度会不会超限
            if length_after_added > self.seq_length - 1:  # end_token保留一个位置
                # save instance
                one_instance = []
                for tokens, end_token, pad_token in zip(instance, ends_tokens, paddings_tokens):
                    tokens.append(end_token)
                    tokens = self.padding(tokens, pad_token)
                    one_instance.append(tokens)

                instances.append(one_instance)  # 加入instances
                # update new instance
                instance = [starts_tokens[i] for i in starts_tokens]

            # append sentence tokens
            for tokens, last_tokens in zip(instance, sentence_tokens):
                tokens.extend(last_tokens)

        # 处理最后一个
        one_instance = []
        for tokens, end_token, pad_token in zip(instance, ends_tokens, paddings_tokens):
            tokens.append(end_token)
            tokens = self.padding(tokens, pad_token)
            one_instance.append(tokens)
        instances.append(one_instance)

        return instances

    def padding(self, seq_tokens, padding_value=None):
        """对单个token序列进行padding"""
        if padding_value is None:
            padding_value = self.token_pad_id

        seq_tokens = seq_tokens[: self.seq_length]
        padding_length = self.seq_length - len(seq_tokens)
        return seq_tokens + [padding_value] * padding_length

    def sentence_process(self, sentence):
        """根据任务对句子进行处理"""
        raise NotImplementedError

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def get_instance_keys(self):
        """Return key of instance's item"""
        raise NotImplementedError

    def tfrecord_serialize(self, instances):
        """
        将instances进行序列化：create_feature -> create_kv_feature -> create_example -> serialize_to_string
        :param instances:
        :param instance_keys: key of instance's item
        :return:
        """
        instance_serialized = []
        instance_keys = self.get_instance_keys()
        for instance in instances:
            features = {k: self._int64_feature(v) for k, v in zip(instance_keys, instance)}
            tf_features = tf.train.Feature(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized = tf_example.SerializeToString()
            instance_serialized.append(serialized)
        return instance_serialized

    @staticmethod
    def load_tfrecord(record_names, batch_size, parse_func):
        """
        读取tfrecord：按parse_func进行parse_single_example -> repeat -> batch
        :param record_names:
        :param batch_size:
        :param parse_func:
        :return:
        """
        if type(record_names) != list:
            record_names = [record_names]

        dataset = tf.data.TFRecordDataset(record_names)  # load
        dataset.map(parse_func)  # pars
        dataset.repeat()  # repeat
        dataset.shuffle()  # shuffle
        dataset.batch(batch_size)  # batch
        return dataset


class TrainingDataSetRoBERTa(TrainingDataset):
    """
    生成Robert模式预训练数据：以词为单位进行mask；15%的概率进行替换；替换时80%替换为 [MASK], 10% 不变，10% 随机替换
    """

    def __init__(self, tokenizer, word_seg, mask_rate=0.15, seq_length=512):
        """

        :param tokenizer:
        :param word_seg: 分词函数
        :param mask_rate:
        :param seq_length:
        """
        super(TrainingDataset, self).__init__(tokenizer, seq_length)
        self.word_seg = word_seg
        self.mask_rate = mask_rate

    def get_instance_keys(self):
        return ['token_ids', 'mask_ids']

    def get_start_end_padding_tokens(self):
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return starts, ends, paddings

    def sentence_process(self, sentence):
        """"""
        words = self.word_seg(sentence)
        probs = np.random.random(len(words))

        sentence_token_ids, sentence_mask_ids = [], []
        for word, prob in zip(words, probs):
            tokens = self.tokenizer.tokenize(words)[1:-1]
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            sentence_token_ids.extend(token_ids)

            if prob < self.mask_rate:
                # token_id加1，让unmask-id为0
                mask_ids = [self.mask_token_process(token_id) + 1 for token_id in token_ids]
            else:
                mask_ids = [0] * len(token_ids)
            sentence_mask_ids.extend(mask_ids)

        return [sentence_token_ids, sentence_mask_ids]

    def mask_token_process(self, token_id):
        """处理mask token ids，其中80%概率替换为 [MASK], 10% 概率不变，10% 概率随机替换"""
        prob = np.random.random()
        if prob < 0.8:
            return self.token_mask_id
        elif prob < 0.9:
            return token_id
        return np.random.randint(0, self.vocab_size)

    @staticmethod
    def load_tfrecord(record_names, seq_length, batch_size):
        def parse_func(serialized_record):
            feature_description = {
                'token_ids': tf.io.FixedLenSequenceFeature([seq_length], tf.int64),
                'mask_ids': tf.io.FixedLenSequenceFeature([seq_length], tf.int64)
            }
            features = tf.io.parse_single_example(serialized_record, feature_description)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(mask_ids, mask_ids - 1, token_ids)  # 之前让位给unmask_id一位，现在减1回归

            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': K.cast(is_masked, K.floatx())
            }
            y = {
                'mlm_loss': K.zeros([1]),
                'mlm_acc': K.zeros([1])
            }
            return x, y

        return TrainingDataset.load_tfrecord(record_names, batch_size, parse_func)


if __name__ == "__main__":
    from toolkit4nlp.tokenizers import Tokenizer
    import re
    import json
    import glob
    import json_fast as jieba
    from tqdm import tqdm

    jieba.initialize()
    vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = Tokenizer(vocab, do_lower_case=True)

    seq_length = 512


    def word_seg(text):
        return jieba.lcut(text)


    def generate_corp():
        file_names = glob.glob('/home/mingming.xu/datasets/NLP/dureader_robust-dataset/pretraining')
        count, sentences = 0, []
        for fname in file_names:
            with open(fname) as fin:
                for p in json.load(fin)['data'][0]['paragraph']:
                    para = [qa['question'] for qa in p]
                    para_text = ' '.join(para)
                    para_text += ' ' + p['context']
                    sentence_list = re.findall('.*?[\n.。 ]+', para_text)
                    sentences.extend(sentence_list)
                    count += 1

                    if count > 10:
                        yield sentences
                        count, sentences = 0, []
        if sentences:
            yield sentences


    dataset = TrainingDataSetRoBERTa(tokenizer=tokenizer, word_seg=word_seg, seq_length=seq_length)

    for i in range(10):  # repeate 10 times to make 10 different ways of mask token
        dataset.process(tqdm(generate_corp()), record_name='../corpus_record/corppus.%i.tfrecord')
