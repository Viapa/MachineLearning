import numpy as np
import time
from collections import Counter, defaultdict
from Utils.metrics import eval_cls_multiple
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_confusion_matrixs

class NaiveBayesClassifer(object):
    """
    目标：实现带有拉普拉斯修正的朴素贝叶斯的分类模型（先验概率为零的处理）
    定义：P(B│A) = P(B) * P(A│B) / P(A) -> P(yi│x1,x2,...,xn) = P(yi) * [P(x1|yi)*P(x2|yi)*...*P(xn|yi)] / [P(x1)*P(x2)*...*P(xn)]
    求解：假定特征之间相互独立，通过训练数据得到X与y的联合分布；之后对于要预测的X，根据贝叶斯公式，输出后验概率最大的yi(argmax(y1,y2...))
    """
    def __init__(self, priors=None, label_set=None, smoothing=1): # 先验概率, 标签集, 平滑因子
        self.priors = priors; # 模型的先验概率, 可以认为指定也可以通过所给数据集估计P(y=Ck)
        if not self.priors:
            self.priors = defaultdict(int); # 若没有指定先验概率，则进行初始化先验概率的字典
        self.label_set = label_set; # 全部可能的标签集（列表或数组形式）
        self.condition = defaultdict(int); # 初始化条件概率字典
        self.smoothing = smoothing;  # 贝叶斯估计中的拉普拉斯平滑系数

    def fit(self, X_data, y_data):
        N = X_data.shape[0] # 输入样本数
        M = X_data.shape[1] # 输入特征数
        K = len(self.label_set) # 标签的类别总数
        # 将全部标签加入到先验概率字典，进行初始化
        for c in self.label_set:
            self.priors[c] = 0
        C_y = Counter(y_data) # 对数据样本的y进行计数
        # 计算先验概率P(y=Ck):
        for key, val in self.priors.items(): # 导出标签不同类别的样本数, key为类别，value为样本数
            self.priors[key] = (C_y[key] + self.smoothing) / (N + K * self.smoothing) # 带拉普拉斯平平滑的先验概率
        # 计算条件概率P(Xj=a|y=Ck):
        for j in range(M): # 对每个特征进行遍历计算
            Xj_y = defaultdict(int) # 建立特征j与y的概率字典
            vector = X_data[:, j] # 取第j个特征
            unique_vector = list(np.unique(vector)) # 特征j的全部取值列表
            Sj = len(unique_vector) # 特征j的全部取值个数
            # 初始化条件概率，包含所有特征与标签的条件情况
            for vector_val in unique_vector:
                for label in self.label_set:
                    Xj_y[(vector_val, label)] = 0
            for xj, y in zip(vector, y_data):  # 统计数据集中特征与标签的条件个数
                Xj_y[(xj, y)] += 1
            for key, val in Xj_y.items():  # 将全部统计加入条件概率字典中
                self.condition[(j, key[0], key[1])] = (val + self.smoothing) / (C_y[key[1]] + Sj * self.smoothing)

        return self.priors, self.condition

    def predict(self, X_test):
        predictions = list() # 建立一个大小相同的初始预测集
        for sample in X_test: # 对每一个样本分别进行预测
            pred_post = dict()  # 建立类别-概率字典, 最后取其中的最大概率对应类别
            for y, prob_y in self.priors.items():  # 导出P(y=Ck)的概率, y为类别，prob_y为每个类别的先验概率
                p_joint = prob_y  # 联合分布概率的第一项（分子的P(y=Ck)）, 利用 log(mn)=log(m)+log(n)代替乘法运算
                for j, Xj in enumerate(sample):  # 导出测试集中的特征序号j和特征名Xj
                    p_joint *= self.condition[(j, Xj, y)]  # 联合分布概率的第二项
                pred_post[y] = p_joint  # 由于分母P(X)相同，无需计算，故直接存储联合概率分布即可判断最大的概率标签
            predictions.append(max(pred_post, key=pred_post.get)) # 返回最大概率值对应的标签名

        return np.array(predictions)


if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Car-Evaluation.csv"
    datasets = load_data(data_dir) # 加载数据, 要求标签必须要混合均匀，每个特征和标签都要有对应
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1]  # 特征
    target = datasets.iloc[:, -1]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.75, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values  # 转化为numpy数组
    start = time.time() # 计算模型的训练时间
    NBC = NaiveBayesClassifer(priors=None, label_set=['unacc', 'acc', 'vgood', 'good'], smoothing=1)  # 实例化模型
    priors, conditions = NBC.fit(X_train, y_train)  # 训练模型
    y_pred = NBC.predict(X_test)  # 预测新数据
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
