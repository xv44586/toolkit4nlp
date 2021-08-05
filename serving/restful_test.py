"""
请求tensorflow-serving 的restful api
环境:
sanic
"""

import json
import numpy as np
import requests

from toolkit4nlp.tokenizers import Tokenizer

from sanic import Sanic
from sanic.response import text, html
from sanic.request import Request


# config
vocab = 'D:/pretrain/chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(vocab, do_lower_case=True)
url = 'http://localhost:8501/v1/models/bert/versions/1:predict'  # model url


def request_result(inputs, url=url):
    token_ids, segment_ids = inputs
    token_ids = token_ids.tolist() if type(token_ids) != list else token_ids
    segment_ids = segment_ids.tolist() if type(segment_ids) != list else segment_ids

    ipt = {'Input-Token': token_ids,
           'Input-Segment': segment_ids,
           }
    request_input = {'inputs': ipt}
    r = requests.post(url, data=json.dumps(request_input)).content
    if type(r) == bytes:
        r = r.decode('utf-8')
    if type(r) == str:
        res = json.loads(r)
        if res and 'outputs' in res:
            ret = res['outputs']
            return np.array(ret)
    return None


def mask_language_model_predict(sentence='科学技术是第一生产力', mask_positions=[2, 3]):
    token_ids, segment_ids = tokenizer.encode(sentence)
    # first token is 'CLS'
    mask_positions = [idx + 1 for idx in mask_positions]
    for idx in mask_positions:
        token_ids[idx] = tokenizer._token_dict['[MASK]']

    probs = request_result([np.array([token_ids]), np.array([segment_ids])])[0]
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
    # return text(pred)
    return html('<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>{}</div>'.format(pred))


@app.route('test', methods=['GET'])
def test(request: Request):
    args = request.args
    sentence = args['sentence'][0]
    idxs = args['idx']
    idxs = [int(idx) for idx in idxs]

    pred = mask_language_model_predict(sentence, idxs)
    return html('<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>{}</div>'.format(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8864', debug=False)

    # 现在打开浏览器输入 http://localhost:8864/test?sentence=科学技术是第一生产力&idx=2&idx=3
    # 或者 http://localhost:8864/test_async?sentence=科学技术是第一生产力&idx=2&idx=3，可以看到对应的结果：技术
