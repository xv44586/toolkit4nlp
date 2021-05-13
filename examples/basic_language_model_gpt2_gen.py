"""
GPT2 对话生成测试
权重文件来自： https://github.com/yangjianxin1/GPT2-chitchat
convert 请参考：https://github.com/xv44586/toolkit4nlp/blob/master/examples/convert_torch_2_tf1.py
"""
import numpy as np

from toolkit4nlp.models import build_transformer_model
from toolkit4nlp.utils import AutoRegressiveDecoder
from toolkit4nlp.tokenizers import Tokenizer

checkpoint_path = 'D:/pretrain/GPT2-Chinese/gpt2.ckpt'
vocab = 'D:/pretrain/GPT2-Chinese/vocab_small.txt'
config = 'D:/pretrain/GPT2-Chinese/config.json'
tokenizer = Tokenizer(vocab, do_lower_case=True)

model = build_transformer_model(config_path=config, checkpoint_path=checkpoint_path, model='gpt2')
model.summary()

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样的文本续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return model.predict(token_ids)[:, -1]

    def response(self, text, n=1, topp=0.95):
        """输出结果会有一定的随机性
        """
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids],
                                     n,
                                     topp=topp)  # 基于随机采样
        # results = [token_ids + [int(i) for i in ids] for ids in results]
        texts = [tokenizer.decode(ids) for ids in results]
        return texts

chatbot = ChatBot(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=16,
)

query = u"""
想看你的美照,
亲我一口就给你看,
我亲两口,
讨厌人家拿小拳拳捶你胸口,
"""
print(chatbot.response(query, 5))
"""
输出结果如： 
['想死', '来啊来啊', '我难道不是这样么', '来来来, 小拳拳锤你胸口', '想想都开心太讨厌了那就看你照片中']
['你死开', '这是你吗', '好恶心！一边儿玩去！', '快去找你的小拳拳锤他胸口', '那我不和你睡了我不跟你睡']
"""