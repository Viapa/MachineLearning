import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from Tools.plotly import plot_scatter, plot_scatter_predict
from Utils.metrics import eval_reg


"""
本节目标: 使用Pytorch和GPU实现对房屋售价的预测
"""

### 数据导入
df = pd.read_csv("D:\python\code\Pytorch\datasets\House_price.csv")
print(df.head())

### 数据分析
plot_scatter(
    x=df['Avg Area Income'].values,
    y=df['Price'].values,
    figsize=(6, 5),
    xlabel="Avg.Area Income",
    ylabel="House Price",
    title="Area vs Price"
)

### 数据预处理/特征工程
X = df.drop('Price', axis=1)  # 特征
X = X.fillna(X.mean())  # 缺失值填充
X = X.values  # 形状(n, k)
Y = df['Price'] / 10000  # 标签
Y = Y.values.reshape(-1, 1) # 形状(n, 1)
print(X.shape, Y.shape)  # 打印形状

# 将特征和标签转为tensor后，放入dataloader
dataloader = DataLoader(TensorDataset(
    torch.tensor(X).float(),
    torch.tensor(Y).float()),
    shuffle = True,
    batch_size = 32
)

### 搭建模型
class APModel(nn.Module):
    def __init__(self):
        super(APModel, self).__init__()  # 继承父类所有方法（例如nn.xx）
        self.linear = nn.Linear(in_features=5, out_features=1, bias=True)  # 建立线性模型，确定输出、输出形状
    def forward(self, x):
        logits = self.linear(x)  # 经过模型后进行输出
        return logits

model = APModel()  # 实例化模型
loss_fn = nn.MSELoss()  # 损失函数选择
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 优化器和学习率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 使用gpu
model.to(device) # 移动模型到cuda
print("If model on cuda:",next(model.parameters()).is_cuda)

### 模型训练
def train(dataloader, epoch):
    start = time.time()
    for i in range(epoch):
        model.train() # 开启训练模式
        for data in dataloader:  # 每次取出一个batch的数据
            x_train, y_train = data
            if torch.cuda.is_available():  # 是否用gpu加载数据
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            y_pred = model(x_train)  # 模型预测
            loss = loss_fn(y_pred, y_train)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器调整参数
        if i % 20 == 0:
            print({"epoch": i, "loss": loss.item()})
    end = time.time()
    print("training done with time used: ", round((end - start), 2))

epoch = 100  # 训练步数
train(dataloader=dataloader, epoch=epoch)  # 开始训练

### 模型预测
X_pred = torch.tensor(X).float().cuda()  # 预测时，数据也要放在和模型同一个device上
y_pred = model(X_pred).detach().cpu().numpy()   # 获取预测值numpy

### 模型评估
plot_scatter_predict(Y, y_pred)
r2 = eval_reg(Y, y_pred, metric='r2')
print(r2)