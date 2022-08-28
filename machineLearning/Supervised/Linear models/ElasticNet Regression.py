import numpy as np
from Utils.metrics import eval_reg
from Utils.datasets import load_data, train_test_split, show_data, Normalize, MinMaxScale
from Utils.plotly import plot_scatter, plot_loss
import math

class ElasticNetRegression(object):
    """
    目标：实现带L1正则和L2正则的弹性网回归模型
    定义：h(x) = w.T * x + b, 损失函数加上 lambda1 * sum(|weights|) + lambda2 * sum(weights^2)
    求解：坐标轴下降，梯度下降法: theta = theta - alpha * [X.T * (h(x) - Y) / m + l1正则项 + l2正则项]
    """
    def __init__(self, alpha=0.01, max_iter=100, l1_ratio=0.5, C=1.0, fit_intercept=True):
        self.alpha = alpha;  # 学习率
        self.max_iter = max_iter;  # 迭代次数
        self.C = C; # 正则项总系数
        self.C1 = l1_ratio * self.C;  # 正则项l1的系数
        self.C2 = (1 - l1_ratio) * self.C # 正则项l2的系数
        self.fit_intercept = fit_intercept;  # 是否添加偏置
        self.weights = None; # 权重参数

    # 定义sign函数
    def sign(self, x):
        if x > 0:
            return 1;
        elif x < 0:
            return -1;
        else:
            return 0

    # 定义参数初始化函数
    def initialize(self, dims, mode="zeros"):
        if mode == "zeros":
            weights = np.zeros((dims, 1))
        elif mode == "ones":
            weights = np.ones((dims, 1))
        elif mode == "uniform":
            limit = 1 / math.sqrt(dims)
            weights = np.random.uniform(-limit, limit, (dims, 1))
        else:
            print("Your initial mode is not supported !")

        return weights

    def l1_l2_loss(self, Xmat, ymat):
        nums = len(Xmat)
        h_x = Xmat * self.weights
        loss = np.sum(np.power(h_x - ymat, 2)) / (2 * nums) + self.C1 * np.sum(np.abs(self.weights)) + self.C2 * np.sum(np.square(self.weights))

        return loss

    def fit(self, X, y):
        lossList = list()
        nums = len(X)
        vec_sign = np.vectorize(self.sign) # 向量化sign函数
        if self.fit_intercept:
            X = np.append(X, np.ones((nums, 1)), axis=1);
        n_features = X.shape[1]  # 计算特征总数（矩阵列）
        Xmat = np.mat(X)  # 矩阵形式特征 m * (n+1)
        self.weights = np.mat(self.initialize(n_features, mode="uniform")) # 初始化权重
        ymat = np.mat(y) # 矩阵形式标签 m * 1
        # 开始迭代训练
        for epoch in range(1, self.max_iter):
            h_x = Xmat * self.weights
            gradient = (Xmat.T * (h_x - ymat)) / nums + self.C1 * vec_sign(self.weights) + self.C2 * self.weights # 计算包含l1+l2的梯度
            self.weights = self.weights - self.alpha * gradient  # 更新权重
            if epoch % 100 == 0:  # 每隔若干次,计算并添加当前损失
                loss = self.l1_l2_loss(Xmat, ymat)
                lossList.append(loss)
                print("epoch %d, loss %f" % (epoch, loss))

        return self.weights, lossList

    def predict(self, X_test):
        nums = len(X_test)  # 新数据样本量
        if self.fit_intercept:
            X_test = np.append(X_test, np.ones((nums, 1)), axis=1);
        feature_mat = np.mat(X_test)
        predictions = np.array(feature_mat * self.weights)

        return predictions

    def get_coef(self): # 获取w系数
        if self.fit_intercept:
            return self.weights[ :-1];
        else:
            return self.weights;

    def get_intercept(self): # 获取截距（如果有）
        if self.fit_intercept:
            return self.weights[-1];
        else:
            print("Model has not intercept !")
            return None


if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Insurance.csv"
    datasets = load_data(data_dir) # 加载数据
    datasets = datasets[:100]  # 选取部分样本
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1] # 特征
    target = datasets.iloc[:, -1] # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.75, seed=2021) # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)  # 转化为numpy数组
    X_train, avg, std = Normalize(X_train) # 训练集数据归一化
    X_test = (X_test - avg) / std # 测试集数据归一化
    ElasticNet = ElasticNetRegression(alpha=0.005, max_iter=2000, l1_ratio=0.8, C=0.5, fit_intercept=True) # 实例化模型
    weights, lossList = ElasticNet.fit(X_train, y_train) # 训练模型
    y_pred = ElasticNet.predict(X_test) # 预测新数据
    print("The models'coefficients are:")
    print(ElasticNet.get_coef())
    print("The models'intercept is:")
    print(ElasticNet.get_intercept())
    print("-------------------Evaluate-------------------")
    y_train, y_test, y_pred = y_train.ravel(), y_test.ravel(), y_pred.ravel() # 将矩阵形状统一为一维（否则计算会出错）
    r2 = eval_reg(y_test.ravel(), y_pred, metric="r2")
    mse = eval_reg(y_test, y_pred, metric="mse")
    rmse = eval_reg(y_test, y_pred, metric="rmse")
    mae = eval_reg(y_test, y_pred, metric="mae")
    mape = eval_reg(y_test, y_pred, metric="mape")
    print("R2 score is: ", r2)
    print("MSE score is: ", mse)
    print("RMSE score is: ", rmse)
    print("MAE score is: ", mae)
    print("MAPE score is: ", mape)
    print("----------------------------------------------")
    print("Scatter plot:")
    plot_scatter(y_test, y_pred)
    print("Loss cruve plot:")
    plot_loss(lossList, color="red")
    print("--------------------END-----------------------")