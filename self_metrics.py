"""
定义各类性能指标
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res

def evaluate(model,data):
    '''

    :param model:
    :param data:
    :return: ACC准确率
    '''
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率: tp + tn / (tp + fp + tn + fn)
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_auc(pred_y, true_y):
    """
    二类别的auc值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    auc = roc_auc_score(true_y, pred_y)
    return auc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算 : tp / (tp+fp)
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率 : tp  / (tp + fn) = sn =recall
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值  f1 = 2 * (recall + pre) / ( recall + pre)
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b

def binary_mcc(pred_y,true_y,positive=1,negative=0):
    '''
     TP： True Positives， 表示实际为正例且被分类器判定为正例的样本数
     FP： False Positives， 表示实际为负例且被分类器判定为正例的样本数
     FN： False Negatives， 表示实际为MCC正例但被分类器判定为负例的样本数
     TN： True Negatives， 表示实际为负例且被分类器判定为负例的样本数
    :param pred_y:
    :param true_y:
    :return:
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(pred_y)):
        if pred_y[i] == positive:  #如果预测为真 ，就=+1，包括TP+FP
            if pred_y[i] == true_y[i]: #如果预测为真，且实际也为真，
                tp += 1
            else:  #预测为真，实际为假，即FP
                fp += 1
        if pred_y[i] == negative: #预测为假 ，即TN+FN
            if pred_y[i] == true_y[i]: #预测为假，实际也为假，即TN
                tn += 1
            else: #预测为假，实际为真 ，即FN
                fn += 1
    try:
        mcc = ((tp*tn) - (fn*fp)) / math.sqrt((tp+fn) * (tn+fp) * (tp+fp) * (tn+fn))
    except:
        mcc = 0
    return mcc


def binary_confusion_matrix(pred_y,true_y,positive=1,negative=0):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(pred_y)):
        if pred_y[i] == positive:  # 如果预测为真 ，就=+1，包括TP+FP
            if pred_y[i] == true_y[i]:  # 如果预测为真，且实际也为真，
                tp += 1
            else:  # 预测为真，实际为假，即FP
                fp += 1
        if pred_y[i] == negative:  # 预测为假 ，即TN+FN
            if pred_y[i] == true_y[i]:  # 预测为假，实际也为假，即TN
                tn += 1
            else:  # 预测为假，实际为真 ，即FN
                fn += 1

    return np.array([tn, fp, fn, tp]).reshape(2,2)

def myMetrics(model,test_gen):
    true_y = []
    pred_y = []
    for x_true, y_true in test_gen:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        true_y.extend(y_true)
        pred_y.extend(y_pred)

    auc, recall, precision, f_beta, mcc, c_matrix = get_binary_metrics(pred_y, true_y)

    return auc, recall, precision, f_beta, mcc, c_matrix

def predict(model, test_gen):
    true_y = []
    pred_y = []
    for x_true, y_true in test_gen:
        y_pred = model.predict(x_true)
        y_true = y_true[:, 0]

        true_y.extend(y_true)
        pred_y.extend(y_pred)
    return true_y, pred_y

def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    # acc = accuracy(pred_y, true_y)
    auc = binary_auc(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    mcc = binary_mcc(pred_y,true_y)
    c_matrix = binary_confusion_matrix(pred_y,true_y)
    return auc, recall, precision, f_beta, mcc, c_matrix


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta