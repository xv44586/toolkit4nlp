# toolkits for NLP

## intent
 为了方便自己学习与理解一些东西，实现一些自己的想法

## Update info:
 -  <strong>2020.08.06</strong> 增加 cws-crf example,具体代码:<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/sequence_labeling_cws_crf.py">cws_crf_example</a>
  - <strong>2020.08.05</strong> 增加 ner-crf example,具体代码:<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/sequence_labeling_ner_crf.py">ner_crf_example</a>
  - <strong>2020.08.01</strong> 增加 bert + dgcnn 做 qa task, 具体代码:<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_dgcnn_example.py">qa_dgcnn_example</a>
  - <strong>2020.07.27</strong> 增加 pretraining，用法参照 <a href="https://github.com/xv44586/toolkit4nlp/blob/master/pretraining/README.md">pretraining/README.md</a>
  - <strong>2020.07.18</strong> 增加 tokenizer， 用法：
  ```python
from toolkit4nlp.tokenizers import Tokenizer
vocab = ''
tokenizer = Tokenizer(vocab, do_lower_case=True)
tokenizer.encode('我爱你中国')    
```
  - <strong>2020.07.16</strong>  完成bert加载预训练权重，用法：
  ```python
from toolkit4nlp.models import build_transformer_model

config_path = ''
checkpoints_path = ''
model = build_transformer_model(config_path, checkpoints_path)
  ```
  
  主要参考了<a href='https://github.com/google-research/bert.git'>bert</a> 和
  <a href='https://github.com/bojone/bert4keras.git'>bert4keras</a>以及
  <a href='https://github.com/CyberZHG/keras-bert'>keras_bert</a>
