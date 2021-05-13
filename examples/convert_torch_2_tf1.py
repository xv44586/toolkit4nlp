#! -*- coding: utf-8 -*-
"""
torch 权重转 tf1 checkpoints，方便后面用toolkit4nlp加载
torch==1.6.0 + tensorflow==1.15.0 + keras==2.3.1
本例将DialoGPT权重转为tf版
"""


import numpy as np

import tensorflow as tf
import torch
import keras.backend as K

weights = torch.load('./DialoGPT-small/pytorch_model.bin', map_location='cpu')
num_hidden_layers = 12

tf_weights = {}
tf_weights['gpt/embeddings/word_embeddings'] = weights['transformer.wte.weight'].numpy()
tf_weights['gpt/embeddings/position_embeddings'] = weights['transformer.wpe.weight'].numpy()

qkv = ['query', 'key', 'value']
for i in range(num_hidden_layers):
    w = weights['transformer.h.%s.attn.c_attn.weight' % i].numpy()
    ws = np.split(w, 3, axis=1)
    for k, w in zip(qkv, ws):
        name = 'gpt/transformer/layer_%s/attention/self/%s/kernel' % (i, k)
        tf_weights[name] = w

    b = weights['transformer.h.%s.attn.c_attn.bias' % i].numpy()
    bs = np.split(b, 3, axis=0)
    for k, b in zip(qkv, bs):
        name = 'gpt/transformer/layer_%s/attention/self/%s/bias' % (i, k)
        tf_weights[name] = b

    # dense
    w = weights['transformer.h.%s.attn.c_proj.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/dense/kernel' % i
    tf_weights[name] = w
    b = weights['transformer.h.%s.attn.c_proj.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/output/dense/bias' % i
    tf_weights[name] = b

    # ln
    w = weights['transformer.h.%s.ln_1.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/input/LayerNorm/gamma' % i
    tf_weights[name] = w
    b = weights['transformer.h.%s.ln_1.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/attention/input/LayerNorm/beta' % i
    tf_weights[name] = b

    w = weights['transformer.h.%s.mlp.c_fc.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/intermediate/dense/kernel' % i
    tf_weights[name] = w
    b = weights['transformer.h.%s.mlp.c_fc.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/intermediate/dense/bias' % i
    tf_weights[name] = b
    w = weights['transformer.h.%s.mlp.c_proj.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/dense/kernel' % i
    tf_weights[name] = w
    b = weights['transformer.h.%s.mlp.c_proj.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/output/dense/bias' % i
    tf_weights[name] = b
    w = weights['transformer.h.%s.ln_2.weight' % i].numpy()
    name = 'gpt/transformer/layer_%s/input/LayerNorm/gamma' % i
    tf_weights[name] = w
    b = weights['transformer.h.%s.ln_2.bias' % i].numpy()
    name = 'gpt/transformer/layer_%s/input/LayerNorm/beta' % i
    tf_weights[name] = b

    # final layer
    w = weights['transformer.ln_f.weight'].numpy()
    name = 'gpt/output/LayerNorm/gamma'
    tf_weights[name] = w
    b = weights['transformer.ln_f.bias'].numpy()
    name = 'gpt/output/LayerNorm/beta'
    tf_weights[name] = b

# save
out_file = './DialoGPT-small/GPT2_model.ckpt'
with tf.Graph().as_default():
    pairs = []
    for name, value in tf_weights.items():
        var = K.variable(tf.zeros(value.shape), name=name)
        pairs.append((var, value))
    with tf.Session() as sess:
        K.batch_set_value(pairs)
        saver = tf.train.Saver()
        saver.save(sess, out_file)