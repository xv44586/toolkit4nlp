# -*- coding: utf-8 -*-
# @Date    : 2020/7/15
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : extract_feature.py
from toolkit4nlp.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np


config = 'D:/pretrain/chinese_L-12_H-768_A-12/bert_config.json'
ckpt = 'D:/pretrain/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab = 'D:/pretrain/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab, do_lower_case=True)

model = build_transformer_model(config, checkpoint_path=ckpt)

tokens, segs = tokenizer.encode('我爱你中国')
print(model.predict([np.array([tokens])], [np.array([segs])]))
'''
[[[-0.44401735 -0.30019525 -0.31233636 ...  0.9529598  -0.4809734
    0.06654366]
  [-0.61762357  0.22118373 -0.07601591 ... -0.17685065 -0.32419267
    0.11741822]
  [-0.7326844  -0.50458074  0.7084515  ...  0.6468811  -0.24373594
    0.15566528]
  ...
  [ 0.19752343  0.21698691 -0.0490396  ...  1.3454045  -0.37932903
    0.54040855]
  [ 0.08867623  0.36128646 -0.10016606 ...  0.7128898  -0.5462814
    0.1091212 ]
  [-0.39198774 -0.03427419 -0.31871173 ...  0.81282985 -0.54821503
    0.01499361]]]
'''
