import numpy as np
from Utils.metrics import eval_cluster
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_cluster, plot_loss
from scipy.spatial.distance import cdist

class K_means_plus(object):
    """
    目标：实现K-means++的优化聚类模型
    定义：解决传统kmeans聚类中由于初始化聚类中心不同而导致模型效果不稳定
    求解：初始化聚类中心时，基于数据分布尽可能使得质心之间距离较远，同时增加n_init功能，确保样本到质心距离之和最小
    """
    def __init__(self, n_clusters=5, max_iter=100, distance="l2", n_init=10, initialize="k-means++"):
        self.n_cluster = n_clusters # 聚类簇数
        self.max_iter = max_iter # 最大迭代次数
        self.initialize = initialize # 初始化质心方式
        self.distance = distance # 计算数据距离的方式
        self.n_init = n_init # 执行n_init次迭代过程，取其距离和最小的迭代作为最终质心位置


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
        original_choice = None # 初始化质心列表
        # 初始化质心位置
        if self.initialize == "random":
            original_choice = np.random.choice(nums, self.n_cluster, replace=False) # 无放回选取K个质心索引
        elif self.initialize == "k-means++":
            first = np.random.choice(nums) # 先选1个作为初始质心索引
            original_choice = [first] # 索引放入选取列表
            all_distances = np.empty((nums, 0)) # 初始化质心距离矩阵（从0维开始添加）
            for i in range(self.n_cluster - 1):  # 继续选取剩下得k-1个初始质心
                center = X[original_choice[i]].reshape(1, -1) # 取出X中得第i个选取的质心
                distances = cdist(X, center, metric=self.dist_func_name) # 计算质心与数据的距离（1维）
                all_distances = np.c_[all_distances, distances]  # 将第i个质心与数据的距离添加到全距离矩阵
                min_distances = all_distances.min(axis=1) # 在全距离矩阵中取距离和最小的质心列
                index = np.argmax(min_distances) # 在最小质心距离列中，选择距离最大的点作为下一个质心（即，离本质心最远处的点）
                original_choice.append(index) # 将其添加到已选质心的索引列表中
        self.centroids = X[original_choice]  # 根据质心列表，从样本中选取该K个质心
        X_cluster = int();
        # 开始迭代训练：计算样本与质心距离，类别标记为质心对应的簇；再次计算质心位置；重复操作到最大迭代
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
                return self.centroids, X_cluster

        return self.centroids, X_cluster

    # 使用迭代优化，取n次随机化首个质心后的最佳状态（可选）
    def fit_opt(self, X):
        lossList = list()  # 存储每次的损失值 J=np.sum((xi-ci)^2)
        centers = list() # 存储每次训练得到的质心
        for i in range(self.n_init):  # 迭代n_init次
            center_i, label_i = self.fit(X) # 第i次训练得到的质心列表和训练集标签
            loss = 0 # 第i次训练的所有类的簇内距离和
            for c in range(self.n_cluster): # 对每个质心与对应标签数据进行距离计算
                c_idx = np.where(label_i == c) # 属于c类的数据索引
                distance_c = np.sum(cdist(X[c_idx], center_i[c].reshape(1, -1))) # c类数据与与c类质心的距离和
                loss += distance_c  # 将c类距离累加到第i次迭代总距离之中
            lossList.append(loss) # 将第i次总距离添加到数组
            centers.append(center_i) # 将第i次质心位置加入数组
        best_idx = int(np.argmin(lossList)) # 在全部n_init次的距离数组中取最小距离为最佳聚类
        self.centroids = centers[best_idx] # 取出相应的聚类中心最为最后结果

        return  self.centroids, lossList

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
    KMeans = K_means_plus(n_clusters=n_cluster, max_iter=200, distance="l2", n_init=8, initialize="k-means++")  # 实例化模型(使用kmeans++后，聚类结果很稳定）
    # centroids, _ = KMeans.fit(X_train)  # 直接训练模型（可能不稳定，因为第一个质心选取是随机的）
    centroids, lossList = KMeans.fit_opt(X_train)  # 迭代训练模型，取最佳质心（非常稳定）
    y_pred = KMeans.predict(X_test)  # 预测新数据(此时预测的簇和原有的标准簇名称可能不一致）
    print("-------------------Evaluate--------------------")
    print(lossList)
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
    print("Distance loss plot:")
    plot_loss(lossList)
    print("Cluster scatter plot:")
    plot_cluster(X_test, y_pred, centroids)
    print("--------------------END-----------------------")