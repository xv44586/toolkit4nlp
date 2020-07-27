# Pretraining
1. 首先改写 process.py 文件，修改自己的文件解析函数
2. 修改pretrain.py 中dataset地址，运行即可

## Background
之前用Bert做fine-tuing就有想过一个问题：由于Bert通常是利用很多个不同领域的语料进行训练的，
而在做下游任务时常常是某个固定领域，如果能在下游任务同领域的语料上训练Bert，结果应该是能提升不少，即使没有额外的语料，
只在样本上做，应该也是能提升一些的。
今年allenai的论文 <a href='https://arxiv.org/pdf/2004.10964.pdf'>Don't stop pretraining</a> 也表示，在领域/任务数据上继续
进行pretraining，能进一步提升性能。