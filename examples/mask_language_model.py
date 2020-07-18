# -*- coding: utf-8 -*-
# @Date    : 2020/7/16
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : mask_language_model.py
import numpy as np

from toolkit4nlp.tokenizers import Tokenizer

from toolkit4nlp.models import build_transformer_model


config = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
ckpt = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab, do_lower_case=True)

model = build_transformer_model(config, checkpoint_path=ckpt, with_mlm=True)

# tokens, segs = tokenizer.encode('北京网聘技术有限公司')
tokens, segs = tokenizer.encode('科学技术是第一生产力')
tokens[3] = tokens[4] = tokenizer._token_dict['[MASK]']

prob = model.predict([np.array([tokens]), np.array([segs])])[0]
print(tokenizer.decode(np.argmax(prob[3:5], axis=1)))
'''
正确结果应该是： 技术
'''