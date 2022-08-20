import numpy as np
from Utils.metrics import eval_reg
from Utils.datasets import load_data, train_test_split, show_data, Normalize, MinMaxScale
from Utils.plotly import plot_scatter

class RidgeRegression(object):
    """
    目标：实现带L2正则项的Ridge回归模型
    定义：h(x) = w.T * x + b, 损失函数加上 lambda * sum(weights^2)
    求解：梯度下降法, 标准方程法: theta = [X.T * X + lambda * E]^-1 * X.T * Y （要求矩阵可逆）
    """
    def __init__(self, C=1.0, fit_intercept=True):
        self.C = C;  # 正则项系数
        self.fit_intercept = fit_intercept;  # 是否添加偏置
        self.weights = None

    def l2_loss(self, Xmat, ymat):
        nums = len(Xmat)
        h_x = Xmat * self.weights
        loss = np.sum(np.power(h_x - ymat, 2)) / (2 * nums) + self.C * np.sum(np.square(self.weights))

        return loss

    def fit(self, X, y):
        nums = len(X)
        if self.fit_intercept:
            X = np.append(X, np.ones((nums, 1)), axis=1);
        n_features = X.shape[1]
        Xmat = np.mat(X)  # 矩阵形式特征 m * (n+1)
        ymat = np.mat(y) # 矩阵形式标签 m * 1
        # 开始求解方程
        xTx = Xmat.T * Xmat
        pre = xTx + self.C * np.eye(n_features)
        suf = Xmat.T * ymat
        self.weights = pre.I * suf
        l2_loss = self.l2_loss(Xmat, ymat)

        return self.weights, l2_loss

    def predict(self, X_test):
        nums = len(X_test);  # 新数据样本量
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
    Ridge = RidgeRegression(C=0.1, fit_intercept=True) # 实例化模型
    weights, loss = Ridge.fit(X_train, y_train) # 训练模型
    y_pred = Ridge.predict(X_test) # 预测新数据
    print("The models'coefficients are:")
    print(Ridge.get_coef())
    print("The models'intercept is:")
    print(Ridge.get_intercept())
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
    print("l2 loss is: ", loss)
    print("----------------------------------------------")
    print("Scatter plot:")
    plot_scatter(y_test, y_pred)
    print("--------------------END-----------------------")