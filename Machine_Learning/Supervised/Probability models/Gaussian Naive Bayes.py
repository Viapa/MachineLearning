import numpy as np
import time
from collections import Counter
from Utils.metrics import eval_cls_multiple
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_confusion_matrixs

class GaussianNB(object):
    """
    目标：实现高斯朴素贝叶斯模型（连续型）
    定义：P(B│A) = P(B) * P(A│B) / P(A) -> P(yi│x1,x2,...,xn) = P(yi) * [P(x1|yi)*P(x2|yi)*...*P(xn|yi)] / [P(x1)*P(x2)*...*P(xn)]
    求解：假定特征之间相互独立，通过训练数据得到X与y的联合分布；其中，条件概率 P(xj|yi) 由yi所在的特征高斯分布 P(x) = 1/sigma√2π * e^[-(x-μ)^2 / 2*var]算出
    """
    def __init__(self, priors=None): # 可设置先验概率
        self.priors = None; # 先验概率
        self.avgs = None; # 特征均值
        self.vars = None; # 特征方差
        self.n_class = None; # 标签类别数

    def _get_prior(self, label): # 计算每个标签的先验概率函数（封装）
        cnt = Counter(label); # 对数据样本的y进行计数
        k = len(cnt) # 样本y的类别数
        prior = [cnt[i] / len(label) for i in range(k)] # 统计P(yi)概率（实际上是用频率代替的）

        return np.array(prior) # 返回数组形式

    def _get_avgs(self, feature, label): # 对每个label类别分别计算特征均值
        avgs = list(); # 保存均值
        for i in range(self.n_class):
            idx = (label == i); # 每一个类别对应的索引
            avg = feature[idx].mean(axis=0) # 计算每个y下特征的均值（列统计每一行数据）
            avgs.append(avg) # 加入到列表中保存

        return np.array(avgs) # 返回数组形式

    def _get_vars(self, feature, label):
        vars = list();  # 保存均值
        for i in range(self.n_class):
            idx = (label == i);  # 每一个类别对应的索引
            var = feature[idx].var(axis=0)  # 计算每个y下特征的方差（列统计每一行数据）
            vars.append(var)  # 加入到列表中保存

        return np.array(vars)  # 返回数组形式

    def _get_likelihood(self, row): # 利用高斯公式，定义计算条件概率（又叫似然度）的map函数，需要进行累乘
        gaussian = 1 / np.sqrt(2 * np.pi * self.vars) * np.exp(-(row - self.avgs)**2 / (2 * self.vars))
        multi_gaussian = gaussian.prod(axis=1) # 对每一行进行累乘

        return multi_gaussian # 返回累乘的条件概率

    def fit(self, X_data, y_data):
        self.priors = self._get_prior(y_data) # 计算先验概率
        self.n_class = len(self.priors) # 计算类别数
        self.avgs = self._get_avgs(X_data, y_data) # 计算均值数组
        self.vars = self._get_vars(X_data, y_data) # 计算方差数组

        return self.priors

    def predict(self, X_test, proba=False):
        likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=X_test) # 使用numpy中的map函数，计算P(xj|yi)
        probs = self.priors * likelihood # 计算联合概率（P(yi) * ΠP(xj|yi))
        probs_sum = probs.sum(axis=1) # 联合概率总分数
        probability = probs / probs_sum[:, None]  # 输出类别概率值，这里[:, None]相当于增加一维
        if proba: # 如果需要输出每个类别的概率
            return probability
        else:
            return probability.argmax(axis=1) # 输出最大概率的标签类别即可


if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Diabetes.csv" # 使用"皮马印第安人糖尿病数据集"进行测试
    datasets = load_data(data_dir) # 加载数据, 要求标签必须要混合均匀，每个特征和标签都要有对应
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1]  # 特征
    target = datasets.iloc[:, -1]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.75, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values  # 转化为numpy数组
    start = time.time() # 计算模型的训练时间
    GNB = GaussianNB()  # 实例化模型
    priors = GNB.fit(X_train, y_train)  # 训练模型
    print(priors)
    y_pred = GNB.predict(X_test)  # 预测新数据
    end = time.time()
    print("Time cost: {:} s".format(np.round(end - start, 6))) # 保留6位小数
    print("-------------------Evaluate-------------------")
    acc = eval_cls_multiple(y_test, y_pred, metric="acc", average="macro")  # 多分类预测
    precision = eval_cls_multiple(y_test, y_pred, metric="precision", average="macro")
    recall = eval_cls_multiple(y_test, y_pred, metric="recall", average="macro")
    f1 = eval_cls_multiple(y_test, y_pred, metric="f1", average="macro")
    print("Accuracy score is: ", acc)
    print("Precision score is: ", precision)
    print("Recall score is: ", recall)
    print("F1 score is: ", f1)
    print("----------------------------------------------")
    print("Confusion-matrix plot:")
    plot_confusion_matrixs(y_test, y_pred)
    print("--------------------END-----------------------")