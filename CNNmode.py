import torch as t
import torchvision
import matplotlib.pyplot as plt
from Module import *
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn想·
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from os.path import join  # 连接路径
import mat4py
import random

# 超参数
batch_size = 25
kernel_size_1 = 30
kernel_size_2 = 50
kernel_num = 32
MaxPool_size = 4
drop_out_rate = 0.1
hidden = 128
learning_rate = 0.005  # 学习率
momentum = 0.5

num_epochs = 100  # 训练次数
log_interval = 10


class Mydataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.train = x_data
        self.label = y_data
        self.transform = transform
        self.len = len(x_data)

    def __getitem__(self, item):
        if self.transform is None:
            return self.train[item], self.label[item]
        return self.Transform(self.train[item]), self.label[item]

    def __len__(self):
        return self.len

    def Transform(self, data):
        pass


input_path = join('.', 'data', 'PPG', 'Training_data')  # 文件路径

train_data = []  # 训练集
train_label = []
test_data = []  # 测试集
test_label = []

for i in range(12):
    id = i + 1
    temp_data = []
    if id == 1:
        file_name = 'DATA_01_TYPE01.mat'
    else:
        if id < 10:
            file_name = 'DATA_0' + str(id) + '_TYPE02.mat'
        else:
            file_name = 'DATA_' + str(id) + '_TYPE02.mat'

    path = join(input_path, file_name)  # 构建文件名
    # print(path)
    data = mat4py.loadmat(path)  # 打开mat文件

    #  取样
    for j in range(2):
        PPG_ori = data['sig'][j + 1]  # 提取文件2，3列（即python列表的第1，2个元素）
        p = 0  # 窗口头
        while p + 1000 - 1 < len(PPG_ori):  # 窗口合法
            temp_data.append((PPG_ori[p:p + 1000], i))  # (x_i,y_i)  形式
            p += 100

    # 划分训练集和测试集
    random.shuffle(temp_data)
    l = len(temp_data)
    train_num = 0.75 * l
    p = 1
    for (x, y) in temp_data:
        if p <= train_num:
            train_data.append(x)
            train_label.append(y)
        else:
            test_data.append(x)
            test_label.append(y)
        p += 1

#  input形式   [ (x_i, y_i)]  其中x_i 为一个列表, y_i为没有进行独热编码的标签
# print(len(train_data))
# print(len(test_data))

train_data = t.Tensor(train_data).view(-1, 1, 1000).float()
train_label = t.Tensor(train_label).long()
test_data = t.Tensor(test_data).view(-1, 1, 1000).float()
test_label = t.Tensor(test_label).long()
# print(train_data.shape)
# print(train_label.shape)
# print(test_data.shape)
# print(test_label.shape)


dataset_train = Mydataset(train_data, train_label)
dataset_test = Mydataset(test_data, test_label)
# print(dataset_train.__len__())
# print(dataset_test.__len__())

train_loader = DataLoader(dataset=dataset_train,
                          batch_size=batch_size,
                          shuffle=True
                          )
test_loader = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=True
                         )
t.cuda.is_available()  # GPU是否可用

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
# print(example_data.shape)
# print(example_data)
# print(example_targets)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Sequential(  # 输入 (batch_size, 1, 1000)
#             nn.Conv1d(
#                 in_channels=1,
#                 out_channels=kernel_num,  # 卷积核数量
#                 kernel_size=kernel_size_1,  # 核尺寸
#                 stride=1,
#             ),  # 输出   (batch_size, 32, 971)
#             nn.BatchNorm1d(num_features=1),
#             nn.SELU(),  # 激活函数
#             nn.MaxPool1d(MaxPool_size),  # 池化核大小4     (batch_size, 32, 242)
#             # nn.Dropout(p=drop_out_rate)  # 采样概率为0.1
#         )
#         self.conv_Block = nn.Sequential(
#
#         )
#
#         self.LSTM1 = nn.LSTM(48, hidden, 1)
#         self.LSTM2 = nn.LSTM(hidden, hidden, 1)
#         self.out = nn.Linear(kernel_num * hidden, 12)
#         self.softmax = nn.LogSoftmax(dim=0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#
#         x, temp = self.LSTM1(x)
#         x, temp = self.LSTM2(x)
#
#         x = x.view(-1, kernel_num * hidden)
#         x = self.out(x)
#
#         x = self.softmax(x)
#
#         return x


network = Net()  # 实例化
if t.cuda.is_available():
    network = network.cuda()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate,  # 优化器，随机梯度下降
#                       momentum=momentum)

optimizer = optim.RMSprop(network.parameters(), lr=learning_rate,  # 文档要求的优化器
                          alpha=0.99, eps=1e-8)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(num_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if t.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = network(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)  # 文档要求的交叉熵loss

        loss.backward()
        optimizer.step()
        '''if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))'''


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:

            if t.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = network(data)
            # test_loss += F.cross_entropy(output, target, size_average=False,reduction='sum').item()  # 交叉熵
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)

        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


for epoch in range(1, num_epochs + 1):
    print("第{}次训练开始！".format(epoch))
    train(epoch)
    test()
