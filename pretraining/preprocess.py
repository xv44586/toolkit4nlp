# -*- coding: utf-8 -*-
# @Date    : 2020/7/20
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : preprocess.py

import tensorflow as tf


class TFRecord(object):
    def __init__(self, tokenizer, seq_length):
        """

        :param tokenizer: tokenizer
        :param seq_length:  sequence length
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.token_pad = tokenizer._token_pad
        self.token_start = tokenizer._token_start
        self.token_sep = tokenizer._token_end
        self.token_mask = tokenizer._token_mask
        self.vocab_size = tokenizer._vocab_size

    def process(self, corpus, record_name, workers=8):
        """将语料（corpus）处理为tfrecord格式，并写入 record_name"""
        writer = tf.io.TFRecordWriter(record_name)
        instances = self.paragraph_process(corpus)
        instances_serialized = self.tfrecord_serialize(instances)
        for instance_serialized in instances_serialized:
            writer.write(instance_serialized)

    def paragraph_process(self, corpus, starts_tokens, ends_tokens, paddings_tokens):
        """
        将句子不断的塞进一个instance里，直到长度即将超出seq_length
        :param corpus: sentence list构成的paragraph
        :param starts_tokens:  每个序列开始的token_id
        :param ends_tokens:   每个序列结束的token_id
        :param paddings_tokens:  每个序列padding_token
        :return: instances
        """
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

    def padding(self, seq_tokens, pad_token=None):
        """对单个token序列进行padding"""
        if pad_token is None:
            pad_token = self.token_pad

        seq_tokens = seq_tokens[: self.seq_length]
        padding_length = self.seq_length - len(seq_tokens)
        return seq_tokens + [pad_token] * padding_length

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
