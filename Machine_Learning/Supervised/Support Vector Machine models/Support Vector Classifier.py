import numpy as np
import time
from Utils.metrics import eval_cls
from Utils.datasets import load_data, train_test_split, show_data
from Utils.plotly import plot_confusion_matrixs

class SVM(object):
    """
    目标：实现支持向量机分类模型
    定义：带有惩罚系数、核函数的支持向量机: wTx + b = 0, min{ ||w||^2 / 2 }, s.t. yi * (wT * xi + b) >= 1
    求解：利用KKT条件、不等式约束的最优化问题,求解凸二次规划的最优化算法，利用SMO算出最优λ值并进行更新。
    """
    def __init__(self, kernel='rbf', C=1.0, tol=1e-3, eps=1e-3):  # 核函数类型, 惩罚系数, 停止训练的误差精度, λ更新阈值
        if kernel not in ["linear", 'rbf']:  # 只支持线性核核高斯核
            raise Exception("Invalid kernel function.")
        if C < 0:  # 惩罚系数不能为负
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        # 变量初始化
        self.kernel = kernel;
        self.C = C;
        self.tol = tol;
        self.eps = eps;
        # 初始化超平面截距b
        self.b = 0

    def kernel_cal(self, x1, x2, gamma="auto"):  # 使用核函数计算x1,x2在高维空间的内积, gamma是高斯核中的参数: K(x1, x2) = exp(-γ*||x1 - x2||²)
        if self.kernel == 'linear':  # 线性核直接相乘
            return np.dot(x1.T, x2)
        elif self.kernel == 'rbf':  # 高斯核利用公式
            if gamma == "auto":  # 自动计算γ参数
                gamma = 1 / self.ncol  # γ=特征个数的倒数
            if not isinstance(gamma, (float, int)):  # 如果γ不是给定的浮点类型, 则报错
                raise TypeError("gamma must be float or int;got(%s)" % type(gamma))
            if gamma <= 0:  # γ也不能小于等于0
                raise ValueError("gamma must be positive; got (gamma = %r)" % gamma)

            return np.exp(-gamma * np.power(np.linalg.norm(x1 - x2), 2)) # 返回高斯核的计算结果

    def label_transform(self, y):  # 检查标签是否为二分类，并将标签转换为-1和+1（必须的）
        self.labels = np.unique(y)  # 标签种类
        if len(self.labels) != 2:  # 若不是两类, 报错
            raise ValueError("The target of dataset is not binary.")
        if set(self.labels) != {-1, 1}:  # 若样本的标签值不为-1, 1
            for i in range(self.nrow):  # 重置每一个样本的标签
                if y[i] == self.labels[0]:
                    y[i] = -1
                else:
                    y[i] = 1

        return y

    def SMO(self, idx_1, idx_2):  # 基于SMO优化算法，选定两个λ进行更新
        if idx_1 == idx_2:
            return 0  # 如果选出的λ索引相同，则优化失败
        alpha1 = self.alpha[idx_1]  # λ1
        alpha2 = self.alpha[idx_2]  # λ2
        y1 = self.y[idx_1]  # y1
        y2 = self.y[idx_2]  # y2
        E1 = self.E[idx_1]  # E1
        E2 = self.E[idx_2]  # E2
        s = y1 * y2  # 简化y1*y2的值
        # 根据y1,y2条件, 计算修剪的上下限L,H
        if y1 != y2:    # L = max(0, λ2(old) - λ1(old)), 高 H = min(C, C + λ2(old) - λ1(old))
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        elif y1 == y2:  # L = max(0, λ2(old) + λ1(old) - C), 高 H = min(C, λ2(old) + λ1(old))
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if L == H:  # 当L等于H,alpha2可取的值确定,此时没有优化空间了,判定为优化失败.
            return 0
        # 计算核函数内积K(xi,xj)
        K11 = self.kernel_cal(self.X[idx_1], self.X[idx_1])
        K22 = self.kernel_cal(self.X[idx_2], self.X[idx_2])
        K12 = self.kernel_cal(self.X[idx_1], self.X[idx_2])
        # 计算alpha2更新公式中 ξ = K11 + K22 - 2*K12 (常量)
        eta = K11 + K22 - 2 * K12
        # 判断ξ是否严格大于0（有时会遇见小于零的情况）
        if eta > 0:  # 满足大于0条件
            a2 = alpha2 + y2 * (E1 - E2) / eta  # λ2(new) = λ2(old) + y2*(E1 - E2) / ξ
            # 裁剪λ2(new)
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:  # 若ξ ≤ 0, 需要对minL求二阶导数，取临界值L1或者H1
            f1 = y1 * (E1 - self.b) - alpha1 * K11 - s * alpha2 * K12
            f2 = y2 * (E2 - self.b) - alpha2 * K22 - s * alpha1 * K12
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha2 + s * (alpha2 - H)
            faiL = L1 * f1 + L * f2 + (1 / 2) * (L1 ** 2) * K11 + (1 / 2) * (L ** 2) * K22 + s * L * L1 * K12 - (L + L1)
            faiH = H1 * f1 + H * f2 + (1 / 2) * (H1 ** 2) * K11 + (1 / 2) * (H ** 2) * K22 + s * H * H1 * K12 - (H + H1)
            # 取两个临界点ΦL,ΦH中的较小者，并给定一点差异范围
            if faiL < faiH - self.eps:
                a2 = L
            elif faiL > faiH + self.eps:
                a2 = H
            else:  # 若L≈H，则更新失败
                a2 = alpha2
        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):  # 若更新后新旧λ2差异不大，也判定更新失败
            return 0
        # 若更新成功, 利用公式更新alpha1的值:  λ1(new) = λ1(old) + y1*y2*(λ2(old) - λ2(new))
        a1 = alpha1 + s * (alpha2 - a2)
        # 更新截距b:
        # (1) if 0 < λ1(new) < C: b1(new) = -E1 - y1*K11*(λ1(new) - λ1(old)) - y2*K21*(λ2(new) - λ2(old)) + b(old)
        # (2) if 0 < λ2(new) < C: b2(new) = -E2 - y1*K12*(λ1(new) - λ1(old)) - y2*K22*(λ2(new) - λ2(old)) + b(old)
        # (3) 如果 λ1(new)， λ2(new) 同时满足上述条件时，则 b1(new) = b2(new)
        # (4) 如果 λ1(new)， λ2(new) 是0或C，则 b1(new), b2(new) 以及它们之间的数都符合KKT条件，此时选取中点作为b(new)
        b1 = -E1 - y1*(a1 - alpha1)*K11 - y2*(a2 - alpha2)*K12 + self.b
        b2 = -E2 - y1*(a1 - alpha1)*K12 - y2*(a2 - alpha2)*K22 + self.b
        if 0 < a1 < self.C:
            # 若a1是支持向量,则用a1更新b1
            self.b = b1
        elif 0 < a2 < self.C:
            # 若a2是支持向量,则用a2更新b2
            self.b = b2
        else: # 否则用中间值更新b
            self.b = (b1 + b2) / 2
        # 将新的λ1,λ2加入到类属性中
        self.alpha[idx_1] = a1
        self.alpha[idx_2] = a2
        # 边界上的样本对应的 λi=0 或者 λi=C，在优化过程中很难变化，然而非边界样本0<λi<C会随着对其他变量的优化会有大的变化
        # 更新非边界样本集所对应的index
        self.non_bound_subset_index = np.argwhere((0 < self.alpha) & (self.alpha < self.C)).flatten()
        # 用新的乘子对self.Ei进行更新
        self.updateEi()

        return 1

    def checkExample(self, idx_2): # 用于检查传入索引对应的样本是否违反KKT条件，因为需要优化的是违反条件的λ
        y2 = self.y[idx_2]   # 通过索引获取对应的标签和拉格朗日乘子λ2
        alpha2 = self.alpha[idx_2]
        # 获取对应E2: E2 = f(x2) - y2
        E2 = self.E[idx_2]
        r2 = y2 * E2  # 残差因子 r2 = y2*E2 = y2*f(x2) - y2² = y2*f(x2) - 1
        # 于是有如下违背KKT条件的判断(外层循环的alpha2必须选择违反KKT条件的支持向量,否则没有更新的必要)
        # 当E1为正时，那么选择最小的Ei作为E2，如果E1为负，选择最大Ei作为E2，通常为每个样本的Ei保存在一个列表中，选择最大的 | E1 - E2 | 来近似最大化步长。
        if ((0 < alpha2) & (r2 > self.tol)) or ((alpha2 < self.C) & (r2 < -self.tol)):
            # 进入循环,说明alpha2已经确定,下面确定alpha1(首先在non-bound subset中寻找alpha1)
            # 若non-bound-subset的元素个数大于1,则使用heuristic的方法在non-bound-subset中确定alpha1
            if self.non_bound_subset_index.shape[0] > 1:
                # E2不论正负,选取E1时,只需找到Ei中最大最小元
                maxEi, max_index = self.E[self.non_bound_subset_index].max(), np.argmax(
                    self.E[self.non_bound_subset_index])
                minEi, min_index = self.E[self.non_bound_subset_index].min(), np.argmin(
                    self.E[self.non_bound_subset_index])
                if abs(maxEi - E2) > abs(minEi - E2):
                    idx_1 = self.non_bound_subset_index[max_index]
                else:
                    idx_1 = self.non_bound_subset_index[min_index]
                if self.SMO(idx_1, idx_2):
                    # 进入SMO部分,尝试更新,若更新成功,则self.SMO返回1,此时应返回1代表更新成功一次.
                    return 1
                # 若通过|E1 - E2|所找到的alpha1更新失败,则遍历non-bound-subset,依次作为alpha1.
                # loop over all non-zero and non-C alpha
                for i in self.non_bound_subset_index:
                    # 若这里的i与上面由启发式算法选出的最优的i1不相等,再更新,否则直接进行下一次尝试。
                    if i != idx_1:
                        idx_1 = i
                        if self.SMO(idx_1, idx_2):
                            # 若更新成功,返回1代表更新成功次数加1.
                            return 1
            # 若在non-bound-subset上全部更新失败或non-bound-subset的支持向量个数少于2个
            # 则遍历整个数据集(除去non-bound-subset)找寻合适的alpha1.
            for idx_1 in range(self.nrow):
                if idx_1 not in self.non_bound_subset_index:
                    if self.SMO(idx_1, idx_2):
                        return 1

        return 0  # 若alpha2不是支持向量或者整个数据集上都没有找到合适的alpha1,返回0代表优化失败.

    def fit(self, X, y):  # 训练函数
        """
        外循环在整个训练集上的单次传递和非边界样本集上的多次传递之间保持交替
        直到整个训练集遵守eps中的KKT条件，算法终止。
        """
        # 获取样本数与特征数
        self.X = X
        self.nrow, self.ncol = self.X.shape
        # 判断样本是否满足二分类的-1,1类型, 并进行转换
        self.y = self.label_transform(y)
        # 初始化拉格朗日乘子（用均匀分布[low,high)中随机采样）
        self.alpha = np.random.uniform(0, self.C, self.nrow)
        # 初始化非边界样本的索引集
        self.non_bound_subset_index = np.array([])
        # 初始化Ei
        # 最开始时由于alpha都初始化为0，所以直接设置预测标签值(target)全部为0
        self.target = self.predict(X)
        self.E = self.target - y
        # 用checkAll变量控制是否遍历整个数据集
        checkAll = True
        # 用numChanged存储一轮循环中优化成功变量的个数
        numChanged = 0
        # 开始循环优化过程
        epoch = 1
        # 外循环
        while checkAll == True or numChanged > 0:
            epoch += 1
            if numChanged > 0:  # 若numChanged >0,说明上一轮while循环中有优化成功的乘子
                numChanged = 0  # 为了该轮循环numChanged能正确反应是否有优化成功的乘子,将numChanged置0.
            if checkAll:  # 如果checkAll 为真,遍历整个数据集
                for i in range(self.nrow):
                    numChanged += self.checkExample(i)
            else:  # 否则，从non-bound subset更新乘子
                for i in self.non_bound_subset_index:
                    numChanged += self.checkExample(i)
            if checkAll == True:  # 若刚刚的遍历是在整个数据集上的进行的
                checkAll = False  # 则让下一次遍历在non-bound subset上进行.
            elif numChanged == 0:  # 若numChanged等于0且checkAll等于False,说明上一轮的while循环在non-bound subset进行的,并且没有乘子更新成功
                checkAll = True  # 则non-bound subset上的所有数据都近似满足KKT条件.下一次遍历应在整个数据集上进行
        # 当退出外循环时: examineAll = True 且 numChanged == 0.
        # 此时说明整个数据集上没有严重违反KKT条件的点了,优化完成.
        print("At {:} epoch, outer-loop has been completed! ".format(epoch))

    def updateEi(self):  # 对每个样本更新误差Ei
        for i in range(self.nrow):
            self.E[i] = self.predict_one(self.X[i]) - self.y[i]

    def predict_one(self, x):  # 预测单个样本的结果
        single_target = 0
        for i in np.argwhere(self.alpha != 0).flatten():
            # 计算new_target时,只需要遍历那些alpha != 0的点即可.
            single_target += self.alpha[i] * self.y[i] * self.kernel_cal(x, self.X[i])

        return single_target + self.b

    def predict(self, X_test):  # 预测新样本集的结果
        nums = len(X_test)  # 待预测样本数
        target = np.zeros(nums)  # 初始化标签为零
        for i in range(nums):  # 对每条数据进行预测
            target[i] = np.sign(self.predict_one(X_test[i]))  # sign函数(-1,+1)

        return target

# 将-1标签映射为0,便于二分类评估
def transform_label(x):
    if x == -1:
        return 0
    else:
        return 1

if __name__ == "__main__":
    data_dir = r"D:/python/code/Machine_Learning/datasets/Breast-Cancer.csv" # 使用"乳腺癌数据集"进行测试
    datasets = load_data(data_dir) # 加载数据, 要求标签必须要混合均匀，每个特征和标签都要有对应
    show_data(datasets, rows=5) # 展示数据
    features = datasets.iloc[:, :-1]  # 特征
    target = datasets.iloc[:, -1]  # 标签
    X_train, X_test, y_train, y_test = train_test_split(features, target, ratio=0.75, seed=2021)  # 切分数据集
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values  # 转化为numpy数组
    start = time.time() # 计算模型的训练时间
    SVC = SVM(kernel='rbf', C=1.0)  # 实例化模型
    SVC.fit(X_train, y_train)  # 训练模型
    y_pred = SVC.predict(X_test)  # 预测新数据
    end = time.time()
    print("Time cost: {:} s".format(np.round(end - start, 6))) # 保留6位小数
    print("-------------------Evaluate-------------------")
    func_vec = np.vectorize(transform_label)  # 数组函数需要向量化
    y_test = func_vec(y_test)
    y_pred = func_vec(y_pred)
    acc = eval_cls(y_test, y_pred, metric="acc")
    precision = eval_cls(y_test, y_pred, metric="precision")
    recall = eval_cls(y_test, y_pred, metric="recall")
    f1 = eval_cls(y_test, y_pred, metric="f1")
    print("Accuracy score is: ", acc)
    print("Precision score is: ", precision)
    print("Recall score is: ", recall)
    print("F1 score is: ", f1)
    print("----------------------------------------------")
    print("Confusion-matrix plot:")
    plot_confusion_matrixs(y_test, y_pred)
    print("--------------------END-----------------------")



