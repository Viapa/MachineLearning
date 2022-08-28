import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve

def plot_scatter(y_true, y_pred): # 绘制回归散点图
    plt.figure(figsize=(6, 5))
    plt.title("Model Preds VS Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Preds")
    plt.scatter(y_true, y_pred, c="green", marker="o")
    plt.plot(np.linspace(np.min(y_true), np.max(y_true), 100),
             np.linspace(np.min(y_pred), np.max(y_pred), 100),
             linestyle='--',
             lw=1.5,
             c='r',
             label='multiple linear model')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def plot_loss(lossList, color="red"):
    plt.plot(lossList, c=color)
    plt.title('Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def plot_roc_auc(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
    plt.plot(fpr, tpr, marker="o")
    plt.show()

def plot_precision_recall(y_true, y_pred_prob):
    plot_precision_recall_curve(y_true, y_pred_prob, figsize=(6, 5))

def plot_confusion_matrixs(y_true, y_pred):
    plot_confusion_matrix(y_true, y_pred, figsize=(6, 5))
    plt.show()

def plot_cluster(X, y_pred, centroids=None, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="o")
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array(range(len(centroids))), s=100)
    plt.title("Clustering result")
    plt.xlabel("V1")
    plt.ylabel("V2")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
