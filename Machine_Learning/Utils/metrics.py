import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, adjusted_mutual_info_score

def eval_reg(y_true, y_pred, metric="r2"): # 回归评估的指标
    if metric == "mse":
        mse = np.sum((y_true - y_pred) ** 2) / len(y_true);
        return mse;
    elif metric == "rmse":
        rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true));
        return rmse
    elif metric == "r2":
        SStot = np.sum((y_true - np.mean(y_true)) ** 2)
        SSres = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - SSres / SStot
        return r2;
    elif metric == "mae":
        mae = np.sum(np.abs(y_true - y_pred)) / len(y_true);
        return mae;
    elif metric == "mape":
        mape = np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_true);
        return mape;
    else:
        print("Your metric has not been defined yet !")

def eval_cls(y_true, y_pred, metric="acc"): # 二分类评估的指标
    def calc_TP(y_true, y_pred):
        tp = 0
        for i, j in zip(y_true, y_pred):
            if i == j == 1:
                tp += 1
        return tp

    def calc_TN(y_true, y_pred):
        tn = 0
        for i, j in zip(y_true, y_pred):
            if i == j == 0:
                tn += 1
        return tn

    def calc_FP(y_true, y_pred):
        fp = 0
        for i, j in zip(y_true, y_pred):
            if i == 0 and j == 1:
                fp += 1
        return fp

    def calc_FN(y_true, y_pred):
        fn = 0
        for i, j in zip(y_true, y_pred):
            if i == 1 and j == 0:
                fn += 1
        return fn

    tp = calc_TP(y_true, y_pred)
    tn = calc_TN(y_true, y_pred)
    fp = calc_FP(y_true, y_pred)
    fn = calc_FN(y_true, y_pred)

    if metric == "acc":
        return (tp + tn) / (tp + tn + fp + fn)
    elif metric == "precision":
        return  tp / (tp + fp)
    elif metric == "recall":
        return tp / (tp + fn)
    elif metric == "f1":
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return 2 * p * r / (p + r)
    elif metric == "auc":
        return roc_auc_score(y_true, y_pred)
    else:
        print("Your metric has not been defined yet !")
        raise TypeError

def eval_cls_multiple(y_true, y_pred, metric="acc", average="macro"): # 多分类评估的指标
    if metric == "acc":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred, average=average)
    elif metric == "recall":
        return recall_score(y_true, y_pred, average=average)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average=average)
    elif metric == "auc":
        return roc_auc_score(y_true, y_pred, average=average, multi_class="ovr")
    else:
        print("Your metric has not been defined yet !")
        raise TypeError

def eval_cluster(X, y_true, y_pred, metric="S-score"):
    if metric == "S-score":
        return silhouette_score(X, y_pred)
    elif metric == "CHI":
        return calinski_harabasz_score(X, y_pred)
    elif metric == "DBI":
        return davies_bouldin_score(X, y_pred)
    elif metric == "ARI":
        return adjusted_rand_score(y_true, y_pred)
    elif metric == "AMI":
        return adjusted_mutual_info_score(y_true, y_pred)
    else:
        print("Your metric has not been defined yet !")
        raise TypeError