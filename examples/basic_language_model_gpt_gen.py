"""
GPT闲聊demo，参考：https://github.com/thu-coai/CDial-GPT
"""
import numpy as np
from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.utils import AutoRegressiveDecoder

config_path = 'D:/pretrain/GPT_LCCC-base-tf/gpt_config.json'
checkpoint_path = 'D:/pretrain/GPT_LCCC-base-tf/gpt_model.ckpt'
dict_path = 'D:/pretrain/GPT_LCCC-base-tf/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
speakers = [
    tokenizer.token_to_id('[speaker1]'),
    tokenizer.token_to_id('[speaker2]')
]

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='gpt',
)

model.summary()

class ChatBot(AutoRegressiveDecoder):
    """
    随机采样生成对话
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states, rtype='probas'):
        token_ids, segment_ids = inputs
        cur_segment_ids = np.zeros_like(output_ids) + token_ids[0, -1]  # which speaker
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, cur_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, n = 3, topk=15, topp=0.9):
        token_ids = [tokenizer._token_start_id, speakers[0]]
        segment_ids = token_ids[:]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i+1) % 2]]  # change to next speaker
            token_ids.extend(ids)
            segment_ids.extend([speakers[i%2]] * len(ids))
            segment_ids[-1] = speakers[(i+1) % 2]  # change to next speaker
        results = self.random_sample([token_ids, segment_ids], n, topk, topp)
        return [tokenizer.decode(rt) for rt in results]

chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
print(chatbot.response(['师傅，你到哪啊？', '到二仙桥。', '该走哪个道啊？', '走成化大道。']))
"""
some responses: 师傅，你去二仙桥了？ /你是哪的啊？/哦哦，好好玩，等你回来/不错不错
"""
