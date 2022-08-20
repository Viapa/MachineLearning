import numpy as np
from Utils.metrics import eval_cls
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_loss, plot_roc_auc

class LogisticRegression(object):
    """
    目标：实现带正则项的逻辑回归（分类）模型
    定义：h(x) = g(w.T * x + b) = 1 / (1 + e^-(w.T * x + b))
    求解：最大似然函数，梯度下降法: theta = theta - alpha * X.T * (h(x) - Y) / m + 正则项
    """
    def __init__(self, penalty="l2", alpha=0.1, max_iter=100, C=1.0, fit_intercept=True):
        self.penalty = penalty; # 正则项
        self.alpha = alpha; # 学习率
        self.max_iter = max_iter; # 迭代次数
        self.C = C; # 正则项系数
        self.fit_intercept = fit_intercept; # 是否添加偏置
        self.weights = None; # 权重参数

    # 计算损失函数J
    def binary_loss(self, X, y):
        nums = len(X) # 样本量
        y_pred = self.sigmoid(X * self.weights) # 预测结果
        loss = (-1 / nums) * np.sum((np.multiply(y, np.log(y_pred)) + np.multiply((1 - y), np.log(1 - y_pred)))) # 二元交叉熵损失

        return loss

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, y):
        lossList = list()  # 记录loss下降过程
        nums = len(X)
        if self.fit_intercept:
            X = np.append(X, np.ones((nums, 1)), axis=1);
        Xmat = np.mat(X) # 矩阵形式特征 m * (n+1)
        n_features = X.shape[1] # 计算特征总数（矩阵列）
        self.weights = np.mat(np.ones((n_features, 1)))  # 初始化权重（矩阵形式） (n+1)*1
        ymat = np.mat(y) # 矩阵形式标签 m * 1
        # 开始迭代训练
        for epoch in range(self.max_iter):
            h_x = self.sigmoid(Xmat * self.weights)
            gradient = (1 / nums) * (Xmat.T * (h_x - ymat)) # 计算梯度
            if self.penalty == "l2": # 正则化梯度(求导后的结果）
                gradient += self.C * self.weights;
            elif self.penalty == "l1":
                gradient += self.C * np.sign(self.weights);
            self.weights = self.weights - self.alpha * gradient # 更新权重
            if epoch % 2 == 0: # 每隔若干次,计算并添加当前损失
                lossList.append(self.binary_loss(Xmat, ymat))

        return self.weights, lossList

    def predict(self, X_test, proba=True):
        nums = len(X_test);  # 新数据样本量
        if self.fit_intercept:
            X_test = np.append(X_test, np.ones((nums, 1)), axis=1);
        feature_mat = np.mat(X_test)
        predictions = self.sigmoid(feature_mat * self.weights)
        if not proba:  # 预测概率/类别
            predictions = list(map(lambda x: 1 if x >= 0.5 else 0, predictions))
            predictions = np.array(predictions)
        else:
            predictions = np.array(predictions).reshape(-1, )

        return predictions

# 主程序入口
if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Iris_2classes.csv"
    datasets = load_data(data_dir) # 加载数据
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1]  # 特征
    target = datasets.iloc[:, -1]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.8, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)  # 转化为numpy数组
    LR = LogisticRegression(penalty="l2", alpha=0.05, max_iter=20, C=0.8, fit_intercept=True)  # 实例化模型
    _, lossList = LR.fit(X_train, y_train)  # 训练模型
    y_pred = LR.predict(X_test, proba=False)  # 预测新数据
    y_pred_prob = LR.predict(X_test, proba=True)
    print("-------------------Evaluate-------------------")
    acc = eval_cls(y_test, y_pred, metric="acc")
    precision = eval_cls(y_test, y_pred, metric="precision")
    recall = eval_cls(y_test, y_pred, metric="recall")
    f1 = eval_cls(y_test, y_pred, metric="f1")
    auc = eval_cls(y_test, y_pred, metric="auc")
    print("Accuracy score is: ", acc)
    print("Precision score is: ", precision)
    print("Recall score is: ", recall)
    print("F1 score is: ", f1)
    print("ROC-AUC score is: ", auc)
    print("----------------------------------------------")
    print("Loss cruve plot:")
    plot_loss(lossList, color="red")
    print("ROC-AUC cruve plot:")
    plot_roc_auc(y_test, y_pred_prob)
    print("--------------------END-----------------------")
