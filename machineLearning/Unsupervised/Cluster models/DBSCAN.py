import numpy as np
from Utils.metrics import eval_cluster
from Utils.datasets import load_data, show_data
from Utils.plotly import plot_cluster, plot_loss

class DBSCAN(object):
    """
    目标：实现基于密度的DBSCAN聚类模型
    定义：解决传统kmeans聚类中无法针对非凸样本集进行良好聚类等问题（可以筛选离群点, 不需要指定k值）
    求解：基于密度的聚类算法, 涉及到两个参数：一个是距离阈值（圆半径ϵ），一个是邻域最少样本数MinPts。
    算法过程：
    1、DBSCAN 需要两个参数：ε (eps) 和形成高密度区域所需要的最少点数 (minPts)，它由一个任意未被访问的点开始，然后探索这个点的 ε-邻域，如果 ε-邻域里有足够的点，则建立一个新的聚类，否则这个点被标签为噪声。注意这个点之后可能被发现在其它点的 ε-邻域里，而该 ε-邻域可能有足够的点，届时这个点会被加入该聚类中。
    2、如果一个点位于一个聚类的密集区域里，它的 ε-邻域里的点也属于该聚类，当这些新的点被加进聚类后，如果它(们)也在密集区域里，它(们)的 ε-邻域里的点也会被加进聚类里。这个过程将一直重复，直至不能再加进更多的点为止，这样，一个密度连结的聚类被完整地找出来。然后，一个未曾被访问的点将被探索，从而发现一个新的聚类或噪声。
    """
    def __init__(self, eps=0.5, MinPts=10, distance="l2"):
        self.eps = eps;
        self.MinPts = MinPts;
        self.distance = distance;

    def calc_distance(self, x1, x2):
        if self.distance == "l1":
            distance = np.abs(x1 - x2)
            distance = np.sum(distance, axis=1)
        elif self.distance == "l2":
            distance = (x1 - x2) ** 2
            distance = np.sqrt(np.sum(distance, axis=1))
        else:
            print("Your distance method is not supported !")
            raise TypeError

        return  distance

    def fit(self, X):
        nums = len(X)
        label = -1 # 初始化标签
        self.X = X # 训练数据必须和预测数据一致，因此先私有化
        self.X_cluster = [-1 for i in range(nums)] # 初始化所有样本的标签为-1
        visited = list() # 记录已访问的样本点
        for idx in range(nums): # 遍历样本
            if idx not in visited: # 当样本没有被访问时，需要计算它与其余样本的距离
                distances = self.calc_distance(X[idx], X)
                NeighborPts = list(np.where(distances <= self.eps)[0]) # 返回小于eps距离的邻域点索引列表
                if len(NeighborPts) < self.MinPts: # 当邻域样本小于样本阈值时,暂时标记为-1（默认值）,后面可能会被加入到其他的邻域内
                    continue;
                else: # 当邻域样本数大于阈值时，需要新增为一个簇类
                    label += 1 # 将标签从0开始更新
                    self.X_cluster[idx] = label # 将该邻域样本设定上标签
                    # 动态添加没有访问的数据点索引到访问列表中
                    for i in NeighborPts:
                        if i not in visited:
                            visited.append(i)
                            new_distances = self.calc_distance(X[i], X) # 计算邻域样本i的数据点距离，找出样本i的邻域样本
                            new_NeighborPts = list(np.where(new_distances <= self.eps)[0])
                            if len(new_NeighborPts) < self.MinPts: # 如果邻域的领域样本数不足，继续循环
                                continue;
                            else: # 如果邻域的领域样本数充足
                                for j in new_NeighborPts: # 找查新邻域中的样本，若不在原邻域样本集里，则进行添加
                                    if j not in NeighborPts:
                                        NeighborPts.append(j)
                            self.X_cluster[i] = label # 将该邻域样本写作同一个簇

    def predict(self, X_test):
        if (self.X == X_test).all(): # 确保训练和预测数据一致，否则无法预测
            predictions = np.array(self.X_cluster)
        else:
            print("Train data and test data must be the same !")
            predictions = np.array()

        return predictions

if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\make_cluster.xlsx"
    datasets = load_data(data_dir)  # 加载数据
    show_data(datasets, rows=5)  # 展示数据
    features = datasets.iloc[:, :-1]  # 聚类特征
    target = datasets.iloc[:, -1]  # 聚类标签
    n_cluster = target.nunique() # 聚类标签的类别数
    X_train, y_train = features.values, target.values # 转化为numpy数组 (DBSCAN中只能一起训练和预测）
    Dbscan = DBSCAN(eps=0.2, MinPts=5, distance="l2")  # 实例化模型（适用于非凸数据集）
    Dbscan.fit(X_train)  # 训练模型，自动分簇
    y_pred = Dbscan.predict(X_train)  # 预测数据(训练数据和预测数据保持一致）
    print(y_pred)
    print("-------------------Evaluate--------------------")
    S_score = eval_cluster(X_train, y_train, y_pred, metric="S-score")  # 聚类指标评估（有标签、无标签）
    CHI = eval_cluster(X_train, y_train, y_pred, metric="CHI")
    DBI = eval_cluster(X_train, y_train, y_pred, metric="DBI")
    ARI = eval_cluster(X_train, y_train, y_pred, metric="ARI")
    AMI = eval_cluster(X_train, y_train, y_pred, metric="AMI")
    print("Silhouette Coefficient is: ", S_score)
    print("Calinski Harabasz Index is: ", CHI)
    print("Davies Bouldin Index is: ", DBI)
    print("Adjusted Rand Index is: ", ARI)
    print("Adjusted Mutual Information is: ", AMI)
    print("----------------------------------------------")
    print("Cluster scatter plot:")
    plot_cluster(X_train, y_pred)
    # print("--------------------END-----------------------")



