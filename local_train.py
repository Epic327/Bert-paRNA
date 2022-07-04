from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from bert_model import Model
from data_helper import load_data
import self_metrics
import plot_curve

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 指定参数
num_classes = 2
maxlen = 50
batch_size = 8
model_select = '2dcnn'

path_prefix = "./bert_weight_files/bert_base_uncased"
config_path = path_prefix + "/bert_config.json"
checkpoint_path = path_prefix + "/bert_model.ckpt"
dict_path = path_prefix + "/vocab.txt"
figure_prefix = './figure/'


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels  # 迭代器，每一次迭代从yield下一行开始
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载数据集
train_data = load_data('./data_csv/train.csv')
valid_data = load_data('./data_csv/valid.csv')
test_data = load_data('./data_csv/test.csv')

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

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

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

if __name__ == '__main__':
    # evaluator = Evaluator()
    checkpointer = ModelCheckpoint(
        filepath='./model/test1.weights',
        verbose=1,
        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=2, mode='min')

    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1,
        callbacks=[checkpointer, earlystopper],
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator)
    )
    plot_curve.lossplot(history, figure_prefix, 'loss11.png')
    print('loss plot finish')
    plot_curve.accplot(history, figure_prefix, 'acc11.png')
    print('acc plot finish')

    model.load_weights('./model/test1.weights')

    print(u'final test acc: %05f\n' % (self_metrics.evaluate(model, test_generator)))
    auc, recall, precision, f_beta, mcc, c_matrix = self_metrics.myMetrics(model, test_generator)
    print('*******************************************************\n')

    print('test auc: %05f' % auc + '\n')
    print('test recall: %05f' % recall + '\n')
    print('test precision: %05f' % precision + '\n')
    print('test f1: %05f' % f_beta + '\n')
    print('test mcc: %05f' % mcc + '\n')
    print('test c_matrix: ', c_matrix)
