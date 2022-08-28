import numpy as np
from Utils.metrics import eval_reg
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_scatter

class MultipleLinearRegression(object):
    """
    目标：实现多元线性回归模型
    定义：h(x) = w.T * x + b
    求解：最小二乘法（要求矩阵可逆，变量不相关）
    """
    def __init__(self, fit_intercept = True): # 初始化参数：是否使用截距b
        self.fit_intercept = fit_intercept;

    def fit(self, X, y):
        self.rows = X.shape[0] # 样本量
        self.cols = X.shape[1] # 样本特征数
        if self.fit_intercept: # 是否使用截距参与运算
            features = np.append(X, np.ones((self.rows, 1)), axis=1);
        else:
            features = X;
        self.__weights = np.dot(np.dot(np.linalg.inv(np.dot(features.T, features)), features.T), y);  # 权重w的计算公式

    def predict(self, X_test):
        nums = len(X_test); # 新数据样本量
        if self.fit_intercept:
            features = np.append(X_test, np.ones((nums, 1)), axis=1);
        else:
            features = X_test
        predictions = np.dot(features, self.__weights) # 预测运算

        return predictions;

    def get_coef(self): # 获取w系数
        if self.fit_intercept:
            return self.__weights[ :-1];
        else:
            return self.__weights;

    def get_intercept(self): # 获取截距（如果有）
        if self.fit_intercept:
            return self.__weights[-1];
        else:
            print("Model has not intercept !")
            return None

# 主程序入口
if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Insurance.csv"
    datasets = load_data(data_dir) # 加载数据
    datasets = datasets[:100]  # 选取部分样本
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1] # 特征
    target = datasets.iloc[:, -1] # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.75, seed=2021) # 切分数据集
    MLR = MultipleLinearRegression(fit_intercept=True) # 实例化模型
    MLR.fit(X_train, y_train) # 训练模型
    y_pred = MLR.predict(X_test) # 预测新数据
    print("The models'coefficients are:")
    print(MLR.get_coef())
    print("The models'intercept is:")
    print(MLR.get_intercept())
    print("-------------------Evaluate-------------------")
    r2 = eval_reg(y_test, y_pred, metric="r2")
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
    print("--------------------END-----------------------")