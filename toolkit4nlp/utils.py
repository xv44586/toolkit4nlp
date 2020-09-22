# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : utils.py
import numpy as np
from abc import abstractmethod


def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


class AutoRegressiveDecoder(object):
    '''
    自回归生成解码器，beam search and random sample两种策略
    '''

    def __init__(self, start_id, end_id, maxlen, minlen=None):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen or 1
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
        else:
            self.first_output_ids = np.array([[self.start_id]])

    def predict(self, inputs, output_ids, states, rtype='probas'):
        '''

        :param inputs:
        :param output_ids:
        :param states:
        :param rtype:logits或probas，用户定义的时候，应当根据rtype来返回不同的结果，
        rtype=probas时返回归一化的概率，rtype=logits时则返回softmax前的结果或者概率对数。
        :return: (scores, states)
        '''
        raise NotImplementedError

    def beam_search(self, inputs, beam_size, states=None, min_ends=1):
        '''

        :param inputs: [ ( seq_length,), ... ]
        :param beam_size:
        :param states:
        :param min_ends: 解码序列中 ends_token_ids 最少次数
        :return: 最优序列
        '''

        inputs = [np.array([i]) for i in inputs]

        output_ids, output_scores = self.first_output_ids, np.zeros(1)

        for step in range(self.maxlen):
            scores, states = self.predict(inputs, output_ids, states, 'logits')
            if step == 0:
                inputs = [np.repeat(i, beam_size, axis=0) for i in inputs]

            scores = output_scores.reshape((-1, 1)) + scores  # 累计得分

            indices = scores.argpartition(-beam_size, axis=None)[
                      -beam_size:]  # flatten array 然后取全局beam_size个最大score的indices
            indices_row = indices // scores.shape[1]  # 行索引， 即对应的路径索引
            indices_col = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引，即token_index
            output_ids = np.concatenate([output_ids[indices_row], indices_col], axis=1)  # 将最大的token_ids 拼到对应路径上
            output_scores = np.take_along_axis(scores, indices, axis=None)  # 更新得分

            ends_counts = (output_ids == self.end_id).sum(1)  # 统计每个路径上的ends_token次数
            if output_ids.shape[1] >= self.minlen:  # 判断路径长度是否达到最短要求
                best_path_idx = output_scores.argmax()  # 得分最高路径
                if ends_counts[best_path_idx] == min_ends:  # 达到最少ends_token要求
                    return output_ids[best_path_idx]  # 返回最优路径
                else:  # 剔除已经结束但是得分不是最高的路径
                    flag = ends_counts < min_ends
                    if not flag.all():
                        inputs = [i[flag] for i in inputs]  # 删除inputs对应路径
                        output_ids = output_ids[flag]  # 删除output对应路径
                        beam_size = flag.sum()  # 重新计算beamsize
                        output_scores = output_scores[flag]
        # 达到长度后直接输出
        return output_ids[output_scores.argmax()]

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含：1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理。
        """

        def actual_decorator(predict):
            def new_predict(
                    self, inputs, output_ids, states, rtype=default_rtype
            ):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)
                if not use_states:
                    prediction = [prediction, None]
                if default_rtype == 'probas':
                    if rtype == 'probas':
                        return prediction
                    else:
                        return np.log(prediction[0] + 1e-12), prediction[1]
                else:
                    if rtype == 'probas':
                        return softmax(prediction[0], -1), prediction[1]
                    else:
                        return prediction

            return new_predict

        return actual_decorator

    def random_sample(self, inputs, n, topk=None, topp=None, states=None, min_ends=1):
        '''
        随机采样生成n个序列
        :param inputs:
        :param n:
        :param topk: 非None 则从概率最高的K个样本中采样
        :param topp:  非None，则从概率逆序排列后累计和不高与topp的token中采样(至少保留一个token供采样）
        :param states:
        :param min_ends: ends token出现的最少次数
        :return:
        '''
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        result = []
        for step in range(self.maxlen):
            probas, states = self.predict(inputs, output_ids, states, 'probas')
            #  第一步时复制n份
            if step == 0:
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk:
                indices_k = probas.argpartition(-topk, axis=1)[:, -topk:]  # 概率最高的K个索引
                probas = np.take_along_axis(probas, indices_k, axis=1)  # 概率最高K个
                probas /= probas.sum(1, keepdims=True)  # 重新归一化概率

            if topp:
                indices_p = probas.argsort(axis=1)[:, ::-1]  # 逆序排列
                probas = np.take_along_axis(probas, indices_p, axis=1)  # 概率逆序排列
                # 概率累计，将大于topp的置零。保证至少一个不为0
                cumsum = np.cumsum(probas, axis=1)
                #                 flag = np.roll(cumsum>=topp, 1, axis=1)
                flag = cumsum >= topp
                flag[:, 0] = False  # 后移一位并将第一位保留，以保证至少一个不为0
                probas[flag] = 0.
                probas /= probas.sum(1, keepdims=True)
            func_sample = lambda p: np.random.choice(len(p), p=p)  # 以概率p随机取一个样本
            sample_ids = np.apply_along_axis(func_sample, axis=1, arr=probas)
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐output_ids
            if topp:
                sample_ids = np.take_along_axis(indices_p, sample_ids, axis=1)
            if topk:
                sample_ids = np.take_along_axis(indices_k, sample_ids, axis=1)
            # 拼接到output
            output_ids = np.concatenate([output_ids, sample_ids], axis=1)
            end_counts = (output_ids == self.end_id).sum(1)  # 统计结束标记
            if output_ids.shape[1] >= self.minlen:  # 判断是否达到最短要求
                flag = end_counts >= min_ends  # 统计满足结束标记结果
                # 收集已结束句子并更新 inputs 和 output
                if flag.any():
                    for i in output_ids[flag]:
                        result.append(i)

                    remain_flag = flag == False
                    inputs = [i[remain_flag] for i in inputs]
                    output_ids = output_ids[remain_flag]
                # 没有剩余句子则跳出
                if len(output_ids) == 0:
                    break
        # 达到最大长度任然没有结束的直接添加进结果列表
        result.extend(output_ids)
        return result


def insert_arguments(**arguments):
    """类的方法上插入一个带有默认值的参数"""

    def decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)  # 用户自定义则覆盖默认值
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return decorator


def remove_arguments(*argments):
    """类方法上禁用某些参数"""

    def decorator(func):
        def new_func(self, *args, **kwargs):
            for k in argments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k))

            return func(self, *args, **kwargs)

        return new_func

    return decorator


class DataGenerator(object):
    """
    数据生成器，用于生成 批量 样本
    example:
    class CIFAR10Generator(DataGenerator):
            def __iter__(self):
                batch_x, batch_y = [], []
                for is_end, item in self.get_sample():
                    file_name, y = item
                    batch_x.append(resize(imread(file_name),(200,200))
                    batch_y.append(y)
                    if is_end or len(batch_x) == self.batch_size:
                        yield batch_x, batch_y
                        batch_x, batch_y = [], []
    cifar10_generate = (file_names_with_label, batch_size=32, shuffle=True)
    """

    def __init__(self, data, batch_size=32, buffer_size=None):
        """
        样本迭代器
        """
        self.data = data
        self.batch_size = batch_size
        if hasattr(data, '__len__'):
            self.steps = int(np.ceil(len(data) / float(batch_size)))
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def get_sample(self, shuffle=False):
        """
        gets one sample data with a flag of is this data is the last one
        """
        if shuffle:
            if self.steps is None:
                def generator():
                    cache, buffer_full = [], False
                    for item in self.data:
                        cache.append(item)
                        if buffer_full:
                            idx = np.random.randint(len(cache))
                            yield cache.pop(idx)
                        elif len(cache) == self.buffer_size:
                            buffer_full = True

                    while cache:
                        idx = np.random.randint(len(cache))
                        yield cache.pop(idx)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for idx in indices:
                        yield self.data[idx]

            data = generator()
        else:
            data = iter(self.data)

        current_data = next(data)
        for next_data in data:
            yield False, current_data
            current_data = next_data

        yield True, current_data

    @abstractmethod
    def __iter__(self, shuffle=False):
        """ 处理单个样本并构造batch data
        """
        raise NotImplementedError

    def generator(self):
        while True:
            for d in self.__iter__(shuffle=True):
                yield d

    def take(self, nums=1):
        """take nums * batch examples"""
        d = []
        for i, data in enumerate(self.__iter__()):
            if i >= nums:
                break

            d.append(data)

        if nums == 1:
            return d[0]
        return d


def pad_sequences(sequences, maxlen=None, value=0):
    """
    pad sequences (num_samples, num_timesteps) to same length
    """
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)

    outputs = []
    for x in sequences:
        x = x[:maxlen]
        pad_range = (0, maxlen - len(x))
        x = np.pad(array=x, pad_width=pad_range, mode='constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class ViterbiDecoder(object):
    """viterbi 解码基类"""

    def __init__(self, trans, starts=None, ends=None):
        """
        :param trans:  转移矩阵
        :param starts:  开始标签index集合
        :param ends: 结束标签index集合
        :return:
        """
        self.trans = trans
        self.num_labels = len(trans)
        self.starts = starts
        self.ends = ends

        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            all_labels = list(range(self.num_labels))
            self.non_starts = [label for label in all_labels if label not in starts]
        if ends is not None:
            all_labels = list(range(self.num_labels))
            self.non_ends = [label for label in all_labels if label not in starts]

    def decode(self, points):
        """points shape: (sequence_length, num_labels)"""
        points[0, self.non_starts] -= np.inf
        points[-1, self.non_ends] -= np.inf
        paths = np.arange(self.num_labels).reshape((-1, 1))
        score = points[0].reshape((-1, 1))
        labels = paths
        for idx in range(1, len(points)):
            all_scores = score + self.trans + points[idx].reshape((1, -1))
            max_idx = all_scores.argmax(0)
            score = all_scores.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[max_idx, :], labels], axis=-1)

        return paths[score[:, 0].argmax(), :]


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """过滤strips，按照seps顺序切分句子为若干个短句子"""
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]
