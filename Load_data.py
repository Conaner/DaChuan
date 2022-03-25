from os.path import join  # 连接路径

import torch.optim as optim


from CNN_LSTM import *
import mat4py
import torch as t
from torch.utils.data import Dataset, DataLoader

# 超参数
batch_size = 64


class Mydataset(Dataset):
    def __init__(self, x_data, y_data):
        self.train = x_data
        self.label = y_data
        self.len = len(x_data)

    def __getitem__(self, item):
        return self.train[item], self.label[item]

    def __len__(self):
        return self.len


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
            temp_data.append((PPG_ori[p:p + 1000], id))  # (x_i,y_i)  形式
            p += 100

    # 划分训练集和测试集
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

train_data = t.Tensor(train_data)
train_label = t.Tensor(train_label)
test_data = t.Tensor(test_data)
test_label = t.Tensor(test_label)

dataset_train = Mydataset(train_data, train_label)
dataset_test = Mydataset(test_data, test_label)

train_loader = DataLoader(dataset=dataset_train,
                          batch_size=batch_size,
                          shuffle=True
                          )
test_loader = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=True
                         )

# examples = iter(train_loader)
# example_data, example_targets = next(examples)

cnn_lstm = cnn_lstm()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = optim.SGD(cnn_lstm.parameters(),lr=learning_rate)
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 10
for i in range(epoch):
    print("第{}轮训练开始".format(i))

    for data,target in train_loader:
        output = cnn_lstm(data)
        loss = loss_fn(output,target)
        print(loss)

