from bert4keras.models import build_transformer_model
import tensorflow as tf

vocab = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_config.json'
checkpint_path = '/home/mingming.xu/pretrain/NLP/chinese_L-12_H-768_A-12/bert_model.ckpt'

bert  = build_transformer_model(config_path=config_path,
                                checkpint_path=checkpint_path, 
                                with_mlm=True)
                                
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(sess, 
                               './bert', 
                               inputs = {'Input-Token': bert.inputs[0], 
                                         'Input-Segment': bert.inputs[1],},
                               outputs={'Output': bert.outputs[0]}
                               
                              )