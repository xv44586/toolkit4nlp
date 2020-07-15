# -*- coding: utf-8 -*-
# @Date    : 2020/7/15
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : extract_feature.py
from toolkit4nlp.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np


config = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
ckpt = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab, do_lower_case=True)

model = build_transformer_model(config, checkpoint_path=ckpt)

token_ids, segment_ids = tokenizer.encode(u'我爱你中国')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
'''[[[-0.00827831  0.5271165  -0.26166633 ...  0.7717155   0.66828513
   -0.34813306]
  [ 0.36656314  0.35970867  0.07721951 ... -0.52110994 -0.4672486
    0.07845952]
  [ 0.69852203 -0.04392128 -1.3160574  ...  1.0618634   0.8293196
    0.07258586]
  ...
  [ 0.25169933  0.30482647 -1.2513853  ...  0.54380953  0.46753684
   -0.6188331 ]
  [ 0.07904248 -0.08373412 -0.39639032 ...  0.29524505  0.748772
   -0.27334788]
  [ 0.2292067   0.10579094  0.38394752 ...  0.60277426  0.02615449
   -0.15034613]]]
'''
