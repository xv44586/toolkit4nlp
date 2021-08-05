"""
sanic 做backend，请求tensorflow-serving 的grpc api

环境：
tensorflow==1.15.0
tensorflow-serving-api==1.10.1
sanic
grpcio==1.26.0
"""
import numpy as np
import grpc

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from sanic import Sanic
from sanic.response import text, html
from sanic.request import Request

from toolkit4nlp.tokenizers import Tokenizer

# config
vocab = '/data/pretrain/chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(vocab, do_lower_case=True)
server = '127.0.0.1:8500'  # grpc api
channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def predict(inputs):
    token_ids, segment_ids = inputs

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'bert'  # model name
    request.model_spec.signature_name = ''  # signature name

    request.inputs['Input-Token'].CopyFrom(tf.make_tensor_proto(token_ids, 'float'))
    request.inputs['Input-Segment'].CopyFrom(tf.make_tensor_proto(segment_ids, 'float'))
    result_future = stub.Predict.future(request, 1.0)  # 10 secs timeout
    result = result_future.result()

    output = np.array(result.outputs['Output'].float_val)
    # result is a list, we need reshape to correct shape
    output = np.reshape(output, token_ids.shape + (-1,))
    return output


def mask_language_model_predict(sentence='科学技术是第一生产力', mask_positions=[2, 3]):
    token_ids, segment_ids = tokenizer.encode(sentence)
    # first token is 'CLS'
    mask_positions = [idx + 1 for idx in mask_positions]
    for idx in mask_positions:
        token_ids[idx] = tokenizer._token_dict['[MASK]']

    probs = predict([np.array([token_ids]), np.array([segment_ids])])[0]
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
    print(pred)
    return html('<!DOCTYPE html><html lang="en"><meta charset="UTF-8"><div>{}</div>'.format(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8864', debug=False)

    # 现在打开浏览器输入 http://localhost:8864/test?sentence=科学技术是第一生产力&idx=2&idx=3
    # 或者 http://localhost:8864/test_async?sentence=科学技术是第一生产力&idx=2&idx=3，可以看到对应的结果：技术
