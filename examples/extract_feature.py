# -*- coding: utf-8 -*-
# @Date    : 2020/7/15
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : extract_feature.py
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.tokenizers import Tokenizer
import numpy as np


config = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
ckpt = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab, do_lower_case=True)

model = build_transformer_model(config, checkpoint_path=ckpt)

token_ids, segment_ids = tokenizer.encode(u'我爱你中国')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
'''[[[-0.00827767  0.52711666 -0.2616654  ...  0.7717162   0.6682844
   -0.3481327 ]
  [ 0.3665638   0.35970846  0.0772187  ... -0.5211092  -0.46724823
    0.07845997]
  [ 0.6985213  -0.04391993 -1.3160559  ...  1.061864    0.8293197
    0.07258661]
  ...
  [ 0.25169933  0.3048255  -1.2513847  ...  0.5438095   0.46753633
   -0.61883307]
  [ 0.07904327 -0.08373377 -0.3963912  ...  0.29524678  0.74877214
   -0.27334687]
  [ 0.22920786  0.10579191  0.38394836 ...  0.60277367  0.02615384
   -0.15034588]]]
'''
