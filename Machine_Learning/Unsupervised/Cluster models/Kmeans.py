import numpy as np
from Utils.metrics import eval_cluster
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_cluster
from scipy.spatial.distance import cdist

class K_means(object):
    """
    目标：实现K-means聚类模型
    定义：对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点距离尽可能小，而让簇间的距离尽可能大。
    求解：循环计算距离，更新质心位置，直到满足收敛情况。
    """
    def __init__(self, n_clusters=5, max_iter=100, distance="l2", initialize="random", centroids_array=None):
        self.n_cluster = n_clusters # 聚类簇数
        self.max_iter = max_iter # 最大迭代次数
        self.initialize = initialize # 初始化质心方式
        self.centroid_array = centroids_array # 外部输入的初始质心
        self.distance = distance # 计算数据距离的方式

    def calc_distance(self):
        if self.distance == "l1":
            return "cityblock"
        elif self.distance == "l2":
            return "euclidean"
        elif self.distance == "mashi":
            return "mahalanobis"
        else:
            return self.distance

    def fit(self, X):
        nums = len(X)
        self.dist_func_name = self.calc_distance();
        # 初始化质心位置
        if self.initialize == "random":
            original_choice = np.random.choice(nums, self.n_cluster, replace=False) # 无放回选取K个质心索引
            self.centroids = X[original_choice] # 从样本中选取该K个质心
        elif self.initialize == "array":
            self.centroids = np.array(self.centroid_array, dtype=np.float)  # 外部输入二维数组的质心位置
        # 迭代训练：计算样本与质心距离，类别标记为质心对应的簇；再次计算质心位置；重复操作到最大迭代
        for epoch in range(self.max_iter):
            # 1.计算距离softmax矩阵（每个样本点到n个质心的距离, [samples * n_clusters]），使用scipy中的.cdist()方法
            distances = cdist(X, self.centroids, metric=self.dist_func_name)
            # 2.对距离进行排序，选取最近邻的质心点类别(即从0开始到K-1的数字索引）作为这些样本点的类别
            X_cluster = np.argmin(distances, axis=1)
            # 3.对每一类数据进行均值计算，更新质心点的坐标
            if epoch > 0:
                original_center = self.centroids.copy()  # 记录原来的质心位置
            else:
                original_center = None;
            for c in range(self.n_cluster):
                # 只更新存在于在X_cluster中的簇质心位置（没有被预测出则不变）
                if c in X_cluster:
                    # 对每个簇的样本重新计算质心，取该簇中所有数据点的均值，作为下一个质心位置（即K-means中的means）
                    c_idx = np.where(X_cluster == c)
                    self.centroids[c] = np.mean(X[c_idx], axis=0)
            # 4.与上一次迭代的质心比较，如果没有发生变化，则提前停止迭代
            if (self.centroids == original_center).all():
                print("The iteration has converged with epoch: {:}".format(epoch))
                return self.centroids;

        return self.centroids

    def predict(self, X_test):
        # 计算距离矩阵，选取距离最近的质心作为该样本的类别
        distances = cdist(X_test, self.centroids, metric=self.dist_func_name)
        predictions = np.argmin(distances, axis=1)

        return predictions

if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\make_cluster.xlsx"
    datasets = load_data(data_dir)  # 加载数据
    show_data(datasets, rows=5)  # 展示数据
    features = datasets.iloc[:, :-1]  # 聚类特征
    target = datasets.iloc[:, -1]  # 聚类标签
    n_cluster = target.nunique() # 聚类标签的类别数
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.6, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values  # 转化为numpy数组
    KMeans = K_means(n_clusters=n_cluster, max_iter=200, distance="cosine", initialize="random")  # 实例化模型（使用random后，聚类结果不稳定）
#    KMeans = K_means(n_clusters=n_cluster, initialize="array", centroids_array=[[-1,2], [0,5], [7,-6], [8,2], [-2,6]])
    centroids = KMeans.fit(X_train)  # 训练模型
    y_pred = KMeans.predict(X_test)  # 预测新数据(此时预测的簇和原有的标准簇名称可能不一致）
    print("-------------------Evaluate--------------------")
    S_score = eval_cluster(X_test, y_test, y_pred, metric="S-score")  # 聚类指标评估（有标签、无标签）
    CHI = eval_cluster(X_test, y_test, y_pred, metric="CHI")
    DBI = eval_cluster(X_test, y_test, y_pred, metric="DBI")
    ARI = eval_cluster(X_test, y_test, y_pred, metric="ARI")
    AMI = eval_cluster(X_test, y_test, y_pred, metric="AMI")
    print("Silhouette Coefficient is: ", S_score)
    print("Calinski Harabasz Index is: ", CHI)
    print("Davies Bouldin Index is: ", DBI)
    print("Adjusted Rand Index is: ", ARI)
    print("Adjusted Mutual Information is: ", AMI)
    print("----------------------------------------------")
    print("Cluster scatter plot:")
    plot_cluster(X_test, y_pred, centroids)
    print("--------------------END-----------------------")