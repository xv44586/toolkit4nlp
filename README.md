# toolkits for NLP

## intent
 为了方便自己学习与理解一些东西，实现一些自己的想法

## Update info:
  - <strong>2020.11.26</strong> 增加pretrain + fine-tuning example, 具体代码：<a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_tnews_pretrain_before_finetuning.py'>classification tnew pretrain before fine-tuning</a>
  - <strong>2020.11.10</strong> NEZHA增加external_embedding_weights, 可以通过该参数将其他信息融合进NEZHA Token-Embedding,具体使用方式：
  ```python
  from toolkit4nlp.models import build_transformer_model
  # 自己构造 embeddings_matrix，与vocabulary 对应
  config_path = ''
  checkpoint_path = ''
  embeddings_matrix = None
  nezha = build_transformer_model(
  config_path=checkpoint_path,
  checkpoint_path=checkpoint_path, 
  model='nezha', external_embedding_size=100,
   external_embedding_weights=embeddings_matrix)
```
  - <strong>2020.10.19</strong> AdaBelief Optimizer 及对应example，具体代码：<a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_adabelief.py'>classification use AdaBelief</a>
  - <strong>2020.10.16</strong> 增加focal loss 及对应example，具体代码：<a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_focal_loss.py'>classification_focal_loss</a>
  - <strong>2020.09.27</strong> 增加NEZHA的实现，使用方法：
  ```python
from toolkit4nlp.models import build_transformer_model
config_path = '/home/mingming.xu/pretrain/NLP/chinese_nezha_base/config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_nezha_base/model_base.ckpt'

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='nezha')
```
  - <strong>2020.09.22</strong> 增加FastBERT的实现，具体代码：<a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_fastbert.py'>classification ifytek with FastBERT</a>
  - <strong>2020.09.15</strong> 增加两个尝试在分类任务上构造新的任务来增强性能实验，具体代码：<a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_with_similarity.py'>classification ifytek with similarity</a> 和 <a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_auxiliary_seq2seq_task.py'>classification ifytek with seq2seq</a>
  - <strong>2020.09.10</strong> 增加Knowledge Distillation Bert example, 具体代码: <a href='https://github.com/xv44586/toolkit4nlp/blob/master/examples/distilling_knowledge_bert.py'>distilling knowledge bert</a>
  - <strong>2020.08.24</strong> 增加UniLM做question answer generation example，具体代码：<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_question_answer_generation_seq2seq.py">qa question answer generation</a>
  - <strong>2020.08.20</strong> 增加UniLM做question generation example，具体代码：<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/qa_question_generation_seq2seq.py">qa question generation</a>
  - <strong>2020.08.20</strong> 增加UniLM和LM model，使用方法：
  ```python
from toolkit4nlp.models import build_transformer_model
config_path = '/home/mingming.xu/pretrain/NLP/chinese_electra_base_L-12_H-768_A-12/config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_electra_base_L-12_H-768_A-12/electra_base.ckpt'


# lm
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    application='lm'
)

# unilm
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    application='unilm'
)

```
  - <strong>2020.08.19</strong> 增加ELECTRA model,使用方法：
  ```python
from toolkit4nlp.models import build_transformer_model


config_path = '/home/mingming.xu/pretrain/NLP/chinese_electra_base_L-12_H-768_A-12/config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_electra_base_L-12_H-768_A-12/electra_base.ckpt'

model =  build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
)

```
  - <strong>2020.08.17</strong> 增加 two-stage-fine-tuning 实验，验证bert-of-theseus中theseus_model的必要性，具体代码: <a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/two_stage_fine_tuning.py">two_stage_fine_tuning</a>
  - <strong>2020.08.14</strong> 增加 bert-of-theseus在ner相关实验下的代码，具体代码：<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/sequence_labeling_ner_bert_of_theseus.py">sequence_labeling_ner_bert_of_theseus</a>
  - <strong>2020.08.11</strong> 增加 bert-of-theseus在文本分类下的相关实验代码，具体代码:<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/classification_ifytek_bert_of_theseus.py">classification_ifytek_bert_of_theseus</a> 
  - <strong>2020.08.06</strong> 增加 cws-crf example,具体代码:<a href="https://github.com/xv44586/toolkit4nlp/blob/master/examples/sequence_labeling_cws_crf.py">cws_crf_example</a>
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
