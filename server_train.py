from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os

import plot_curve
from data_helper import load_data
import self_metrics
from bert_model import Model

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--maxlen', type=int, default=512, help='sentence length in train files')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--out', type=str, default="model.weights")
parser.add_argument('--model', type=str)
parser.add_argument('--tra', type=str)
parser.add_argument('--val', type=str)
parser.add_argument('--loss', type=str)
parser.add_argument('--acc', type=str)

args = parser.parse_args()

epochs = args.epochs
num_classes = args.num_classes
maxlen = args.maxlen
batch_size = args.batch_size
out = args.out
model_select = args.model
tra_data = args.tra
val_data = args.val
loss = args.loss
acc = args.acc

outpath_prefix = "/root/vision/model/bert-seq/"
path_prefix = "/root/vision/weight/ch-bert-base/"
seq_prefix = '/root/vision/data/seq_data/'
figure_prefix = '/root/vision/figure/'

# 预训练模型目录
config_path = path_prefix + "bert_config.json"
checkpoint_path = path_prefix + "bert_model.ckpt"
dict_path = path_prefix + "vocab.txt"


class data_generator(DataGenerator):
    """数据生成器,需要token_ids,segment_ids,labels,
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):

            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)  # 分词器处理

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            # 如果batch_token_id的长度等于batch_size，或者 is_end=true
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载数据集
train_data = load_data(seq_prefix + tra_data)
valid_data = load_data(seq_prefix + val_data)
test_data = load_data(seq_prefix + 'ind_test.csv')

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

bert_model = Model()

if model_select == 'base':
    model = bert_model.bert_dense(config_path, checkpoint_path, num_classes)
elif model_select == 'textcnn':
    model = bert_model.bert_text_cnn(config_path, checkpoint_path, num_classes)
elif model_select == '1dcnn':
    model = bert_model.bert_1d_cnn(config_path, checkpoint_path, num_classes)
elif model_select == '2dcnn':
    model = bert_model.bert_2d_cnn(config_path, checkpoint_path, num_classes)

model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
    #     1000: 1,
    #     2000: 0.1
    # }),
    metrics=['accuracy'],
)

if __name__ == '__main__':
    checkpointer = ModelCheckpoint(
        filepath=outpath_prefix + out,
        verbose=1,
        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=2, mode='min')

    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[checkpointer, earlystopper],
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator)
    )

    plot_curve.lossplot(history, figure_prefix, loss)
    print('loss plot finish')

    plot_curve.accplot(history, figure_prefix, acc)
    print('acc plot finish')

    model.load_weights(outpath_prefix + out)  # 加载模型

    auc, recall, precision, f_beta, mcc, c_matrix = self_metrics.myMetrics(model, valid_generator)
    print('val auc: %05f' % auc + '\n')
    print('val recall: %05f' % recall + '\n')
    print('val precision: %05f' % precision + '\n')
    print('val f1: %05f' % f_beta + '\n')
    print('val mcc: %05f' % mcc + '\n')
    print('val c_matrix: ', c_matrix)

    print('*******************************************************\n')

    auc, recall, precision, f_beta, mcc, c_matrix = self_metrics.myMetrics(model, test_generator)
    print(u'final test acc: %05f\n' % (self_metrics.evaluate(model, test_generator)))
    print('test auc: %05f' % auc + '\n')
    print('test recall: %05f' % recall + '\n')
    print('test precision: %05f' % precision + '\n')
    print('test f1: %05f' % f_beta + '\n')
    print('test mcc: %05f' % mcc + '\n')
    print('test c_matrix: ', c_matrix)
