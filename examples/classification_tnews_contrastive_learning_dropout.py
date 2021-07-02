"""
用dropout 做数据增强，构造样本的不同view，来通过增加对比学习增强分类模型的性能

"""
import json
from tqdm import tqdm

from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import *
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *
from keras.losses import kullback_leibler_divergence as kld

num_classes = 16
maxlen = 64
batch_size = 72

epochs = 5
# BERT base
config_path = '/data/pretrain/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/pretrain/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/pretrain/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label, label_des = l['sentence'], l['label'], l['label_desc']
            label = int(label) - 100 if int(label) < 105 else int(label) - 101
            D.append((text, int(label), label_des))
    return D


# 加载数据集
train_data = load_data(
    '/data/datasets/NLP/CLUE/tnews_public/train.json'
)
valid_data = load_data(
    '/data/datasets/NLP/CLUE/tnews_public/dev.json'
)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    vocab_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label, label_des) in self.get_sample(shuffle):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(data=train_data, batch_size=batch_size)
valid_generator = data_generator(data=valid_data, batch_size=batch_size)


class TotalLoss(Loss):
    "计算两部分loss：分类交叉熵和对比loss"

    def compute_loss(self, inputs, mask=None):
        loss = self.compute_loss_of_classification(inputs)
        sim_loss = self.compute_loss_of_similarity(inputs)
        return loss + sim_loss

    def compute_loss_of_classification(self, inputs, mask=None):
        _, _, y_true, _, y_pred = inputs
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        self.add_metric(loss, 'cls_loss')
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        self.add_metric(loss, 'sim_loss')
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels

    def compute_kld(self, inputs, alpha=4, mask=None):
        _, _, _, y_pred = inputs
        loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        loss = K.mean(loss) / 4 * alpha
        self.add_metric(loss, 'kld')
        return loss


bert = build_transformer_model(checkpoint_path=checkpoint_path,
                               config_path=config_path,
                               keep_tokens=keep_tokens,
                               dropout_rate=0.3,
                               )

label_inputs = Input(shape=(None,), name='label_inputs')

pooler = Lambda(lambda x: x[:, 0])(bert.output)
x = Dense(units=num_classes, activation='softmax', name='classifier')(pooler)
output = TotalLoss(0)(bert.inputs + [label_inputs, pooler, x])

model = Model(bert.inputs + [label_inputs], output)
classifier = Model(bert.inputs, x)

model.compile(optimizer=Adam(2e-5), metrics=['acc'])
model.summary()


def evaluate(val_data=valid_generator):
    total = 0.
    right = 0.
    for (x, s, y_true), _ in tqdm(val_data):
        y_pred = classifier.predict([x, s]).argmax(axis=-1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    print(total, right)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_path='best_model.weights'):
        self.best_val_acc = 0.
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.save_path)

        print('current acc :{}, best val acc: {}'.format(val_acc, self.best_val_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])