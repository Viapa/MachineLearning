import numpy as np
from Utils.datasets import load_data, show_data, train_test_split
from Utils.metrics import eval_cls_multiple


class KNearestNeighbor(object):
    """
    目标：实现K近邻分类模型
    定义：计算样本与训练数据的特征距离，取最近的k个数据对应标签作为预测值
    求解：循环地计算距离即可。增加权重的概念，距离待预测样本点更近的数据，应该获得更大的权重，更远的数据应该获得更小的权重。
    """
    def __init__(self, n_neighbors=3, distance="l1", weights=None):
        self.neighbors = n_neighbors;
        self.distance = distance;
        self.weights = weights;

    def l1_distance(self, x1, x2):
        distance = np.abs(x1 - x2)
        distance = np.sum(distance, axis=1)

        return distance

    def l2_distance(self, x1, x2):
        distance = (x1 - x2) ** 2
        distance = np.sqrt(np.sum(distance, axis=1))

        return  distance

    # 距离权重定义的函数
    def weight_func(self, distance, mode="uniform"):
        if mode == "uniform":
            epsilon = 1e-2
            weight = 1 / (distance + epsilon)
        elif mode == "sigmoid":
            weight = 1 + np.exp(-distance)
        elif mode == "normal":
            weight = -1 * (distance - np.mean(distance)) / np.std(distance)
        else:
            print("Your weight method is not supported !")
            raise TypeError

        return weight

    def fit(self, X, y):
        self.X_train = X;
        self.y_train = y;

    def predict(self, X_test):
        nums = len(X_test)
        predictions = np.zeros((nums, 1), dtype=self.y_train.dtype) # 初始化预测分类的数组
        # 遍历数据点，求取距离排序
        for idx, x in enumerate(X_test):
            # 计算x与所有训练数据的距离
            if self.distance == "l1":
                distances = self.l1_distance(self.X_train, x)
            elif self.distance == "l2":
                distances = self.l2_distance(self.X_train, x)
            else:
                print("Your distance method is not supported !")
                raise TypeError;
            # 将距离按低到高排序，输出索引
            sort_index = np.argsort(distances)
            # 取最近邻的k个点，保存其类别
            k_index = sort_index[: self.neighbors]
            y_label = self.y_train[k_index].ravel()
            k_distances = distances[k_index]
            # 不考虑权重时，统计类别中出现频率最高的作为标签
            if not self.weights:
                y_count = np.bincount(y_label)
                predictions[idx] = np.argmax(y_count)
            # 考虑权重时，统计权重量，取最大权重对应的作为标签
            else:
                y_weight = self.weight_func(k_distances, mode=self.weights)
                label_dict = dict()
                for item in y_label:
                    if item not in label_dict:
                        label_dict.setdefault(item, y_weight[item])
                    else:
                        label_dict[item] += y_weight[item]
                predictions[idx] = max(label_dict, key=label_dict.get)

        return predictions


if __name__ == "__main__":
    data_dir = "D:\python\code\Machine_Learning\datasets\Iris_3classes.csv"
    datasets = load_data(data_dir) # 加载数据
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1]  # 特征
    target = datasets.iloc[:, -1]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.7, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)  # 转化为numpy数组
    KNN = KNearestNeighbor(n_neighbors=7, distance="l2", weights="sigmoid")  # 实例化模型
    KNN.fit(X_train, y_train)  # 训练模型
    y_pred = KNN.predict(X_test)  # 预测新数据
    print("-------------------Evaluate--------------------")
    y_train, y_test, y_pred = y_train.ravel(), y_test.ravel(), y_pred.ravel() # 将矩阵形状统一为一维（否则计算会出错）
    acc = eval_cls_multiple(y_test, y_pred, metric="acc", average="macro") # 多分类预测，需要指定平均计算方式("macro","micro")
    precision = eval_cls_multiple(y_test, y_pred, metric="precision", average="macro")
    recall = eval_cls_multiple(y_test, y_pred, metric="recall", average="macro")
    f1 = eval_cls_multiple(y_test, y_pred, metric="f1", average="macro")
    print("Accuracy score is: ", acc)
    print("Precision score is: ", precision)
    print("Recall score is: ", recall)
    print("F1 score is: ", f1)
    print("--------------------END-----------------------")