import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter(x, y, figsize=(6, 5), xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c="b", marker="o")
    plt.show()

def plot_scatter_predict(y_true, y_pred): # 绘制回归散点图
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

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()