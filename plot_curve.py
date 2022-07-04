import matplotlib.pyplot as plt
from sklearn import metrics
from bert_model import Model
import data_helper as dh
import self_metrics as sm

def lossplot(history,figure_prefix,plot_name):
    fig = plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(figure_prefix+plot_name)


def accplot(history,figure_prefix,plot_name):
    fig = plt.figure()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc, label='accuracy')
    plt.plot(val_acc, label='val_accuracy')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(figure_prefix+ plot_name)





def roc_curve():

    bert_model = Model()
    path_prefix = "/root/vision/weight/ch-bert-base/"
    config_path = path_prefix + "bert_config.json"
    checkpoint_path = path_prefix + "bert_model.ckpt"
    num_classes = 2
    Font = {'size': 18, 'family': 'Times New Roman'}

    test_data = dh.load_data('/root/vision/code/ind_test.csv')

    base_model = bert_model.bert_dense(config_path, checkpoint_path, num_classes)
    cnn1d_model = bert_model.bert_1d_cnn(config_path, checkpoint_path, num_classes)
    cnn2d_model = bert_model.bert_2d_cnn(config_path, checkpoint_path, num_classes)
    textcnn_model = bert_model.bert_text_cnn(config_path, checkpoint_path, num_classes)
    base_model.summary()
    print('****************************')
    cnn1d_model.summary()
    print('****************************')
    cnn2d_model.summary()
    print('****************************')
    textcnn_model.summary()

    base_model.load_weights('/root/vision/code/model/bert-base_1.weights')
    cnn1d_model.load_weights('/root/vision/code/model/bert-1dcnn_1.weights', by_name=True)
    cnn2d_model.load_weights('/root/vision/code/model/bert-2dcnn_2.weights')
    textcnn_model.load_weights('/root/vision/code/model/bert-textcnn_3.weights')

    lw = 2
    test_gen = dh.data_generator(test_data, 32)

    y_test1, y_pred1 = sm.predict(base_model,test_gen)
    y_test1, y_pred2 = sm.predict(cnn1d_model,test_gen)
    y_test1, y_pred3 = sm.predict(cnn2d_model,test_gen)
    y_test1, y_pred4 = sm.predict(textcnn_model,test_gen)


    fpr1, tpr1, thres1 = metrics.roc_curve(y_test1, y_pred1)
    fpr2, tpr2, thres2 = metrics.roc_curve(y_test1, y_pred2)
    fpr3, tpr3, thres3 = metrics.roc_curve(y_test1, y_pred3)
    fpr4, tpr4, thres4 = metrics.roc_curve(y_test1, y_pred4)


    roc_auc1 = 0.917196
    roc_auc2 = 0.906877
    roc_auc3 = 0.911505
    roc_auc4 = 0.915573


    plt.figure(figsize=(10, 10))

    plt.plot(fpr1, tpr1, 'b', label='Bert-Base(AUC = %0.4f)' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label='Bert-1DCNN(AUC = %0.4f)' % roc_auc2, color='darkorange')
    plt.plot(fpr3, tpr3, 'b', label='Bert-2DCNN(AUC = %0.4f)' % roc_auc3, color='green')
    plt.plot(fpr4, tpr4, 'b', label='Bert-1DTextCNN(AUC = %0.4f)' % roc_auc4, color='RoyalBlue')

    plt.legend(loc='lower right', prop=Font)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    #     plt.tick_params(labelsize=15)
    plt.title('eRNAs ROC Curve', Font)
    plt.tick_params(labelsize=15)
    #     plt.show()
    plt.savefig("/root/vision/code/figure/roc-curve3.png")

if __name__ == '__main__':
    roc_curve()