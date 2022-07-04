from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
import argparse
import bert_model
from data_helper import load_data
import self_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--weight', type=str)
parser.add_argument('--maxlen', type=int)
parser.add_argument('--batch', type=int)
parser.add_argument('--model', type=str)

args = parser.parse_args()

weight = args.weight
maxlen = args.maxlen
batch_size = args.batch
model_select = args.model

outpath_prefix = "/root/vision/model/bert-seq/"
path_prefix = "/root/vision/weight/ch-bert-base/"
seq_prefix = '/root/vision/data/seq_data/'
model_prefix = '/root/vision/model/bert-seq/'

# 预训练模型目录
config_path = path_prefix + "bert_config.json"
checkpoint_path = path_prefix + "bert_model.ckpt"
dict_path = path_prefix + "vocab.txt"

tokenizer = Tokenizer(dict_path, do_lower_case=True)

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


if __name__ == '__main__':

    test_data = load_data(seq_prefix + 'test.csv')
    test_generator = data_generator(test_data, batch_size)

    bert_model = bert_model()

    if model_select == 'base':
        model = bert_model.bert_dense(config_path, checkpoint_path, num_classes)
    elif model_select == 'textcnn':
        model = bert_model.bert_text_cnn(config_path, checkpoint_path, num_classes)
    elif model_select == '1dcnn':
        model = bert_model.bert_1d_cnn(config_path, checkpoint_path, num_classes)
    elif model_select == '2dcnn':
        model = bert_model.bert_2d_cnn(config_path, checkpoint_path, num_classes)

    model.summary()

    model.load_weights(model_prefix + weight)

    auc, recall, precision, f_beta, mcc, c_matrix = self_metrics.myMetrics(model, test_generator)

    print('test auc: %05f' % auc + '\n')
    print('test recall: %05f' % recall + '\n')
    print('test precision: %05f' % precision + '\n')
    print('test f1: %05f' % f_beta + '\n')
    print('test mcc: %05f' % mcc + '\n')
    print('test c_matrix: ', c_matrix)
