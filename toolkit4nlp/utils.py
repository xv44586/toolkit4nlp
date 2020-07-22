# -*- coding: utf-8 -*-
# @Date    : 2020/6/29
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : utils.py
import numpy as np


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
