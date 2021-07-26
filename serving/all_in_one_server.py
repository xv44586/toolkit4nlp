"""
直接在server 端load model 然后进行推理
server 框架选择使用sanic，是因为sanic是一个异步框架，相对flask 这类框架性能更好
TIPS: 由于tf 的计算是在Graph的session中，所以需要对每个模型维护自己的graph和session
"""
import numpy as np

import tensorflow as tf
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.backend import K

from sanic import Sanic
from sanic.response import text, html
from sanic.request import Request

checkpoint_path = '/data/pretrain/chinese_L-12_H-768_A-12/bert_model.ckpt'
config_path = '/data/pretrain/chinese_L-12_H-768_A-12/bert_config.json'
vocab = '/data/pretrain/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(vocab, do_lower_case=True)


# builder infer model
class InferModel(object):
    def __init__(self, model_builder):
        self.graph = tf.Graph()
        self.set_session = K.set_session

        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.model = model_builder()

    def predict(self, *args, **kwargs):
        with self.graph.as_default():
            self.set_session(self.sess)
            return self.model.predict(*args, **kwargs)


def bert_builder():
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        with_mlm=True,
    )
    return bert


bert_infer = InferModel(bert_builder)


# mask lannguage model test
def mask_language_model_predict(sentence='科学技术是第一生产力', mask_positions=[2, 3]):
    token_ids, segment_ids = tokenizer.encode(sentence)
    # first token is 'CLS'
    mask_positions = [idx + 1 for idx in mask_positions]
    for idx in mask_positions:
        token_ids[idx] = tokenizer._token_dict['[MASK]']

    probs = bert_infer.predict([np.array([token_ids]), np.array([segment_ids])])[0]
    preds = tokenizer.decode(np.argmax(probs[mask_positions], axis=1))
    return preds


# server
app = Sanic(__name__)


@app.route('test_async', methods=['GET'])
async def test_async(request: Request):
    args = request.args
    sentence = args['sentence'][0]
    idxs = args['idx']
    idxs = [int(idx) for idx in idxs]
    pred = await request.app.loop.run_in_executor(
        None,
        mask_language_model_predict,
        sentence,
        idxs
    )
    # pred = mask_language_model_predict(sentence, idxs)
    # return text(pred)
    return html('<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>{}</div>'.format(pred))


@app.route('test', methods=['GET'])
def test(request: Request):
    args = request.args
    sentence = args['sentence'][0]
    idxs = args['idx']
    idxs = [int(idx) for idx in idxs]

    pred = mask_language_model_predict(sentence, idxs)
    # return text(pred)
    return html('<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>{}</div>'.format(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8864', debug=False)

    # 现在打开浏览器输入 http://localhost:8864/test?sentence=科学技术是第一生产力&idx=2&idx=3
    # 或者 http://localhost:8864/test_async?sentence=科学技术是第一生产力&idx=2&idx=3，可以看到对应的结果：技术
