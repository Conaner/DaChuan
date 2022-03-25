from torch import nn
import torch
import torch.nn.functional as F

# # 超参数
# batch_size = 25
kernel_size_1 = 30
kernel_size_2 = 50
kernel_num = 32
MaxPool_size = 4
# drop_out_rate = 0.1
hidden = 128
# learning_rate = 0.01  # 学习率
# momentum = 0.5
#
# num_epochs = 20  # 训练次数
# log_interval = 10


class Block(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # 输入通道、输出通道、stride、下采样
        super(Block, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=2, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=2, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=2, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.selu = nn.SELU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积操作
        out = self.selu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.selu(self.bn2(self.conv2(out)))
        # print(out.shape)
        out = self.bn3(self.conv3(out))
        # print(out.shape)
        # 是否直连（如果时Identity block就是直连；如果是Conv Block就需要对残差边进行卷积，改变通道数和size）
        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out:',out.shape)
        # print('residual:',residual.shape)
        # torch.reshape(residual,out.shape)
        # 将参差部分和卷积部分相加
        out += residual
        out = self.selu(out)

        return out


class Net(nn.Module):
    def __init__(self, kernel_num=kernel_num, layers=None):
        super(Net, self).__init__()
        if layers is None:
            layers = [2, 3]
        self.kernel_num = kernel_num
        # self.block = block
        self.conv = nn.Sequential(  # 输入 (batch_size, 1, 1000)
            nn.Conv1d(
                in_channels=1,
                out_channels=kernel_num,  # 卷积核数量
                kernel_size=kernel_size_1,  # 核尺寸
                stride=1,
            ),  # 输出   (batch_size, 32, 971)
            nn.BatchNorm1d(num_features=kernel_num),  # ...
            nn.SELU(),  # 激活函数
            nn.MaxPool1d(MaxPool_size),  # 池化核大小4     (batch_size, 32, 242)
            # nn.Dropout(p=drop_out_rate)  # 采样概率为0.1
        )
        #

        self.Block1 = self.make_layer(Block, 64, layers[0], stride=1)
        self.Block2 = self.make_layer(Block, 32, layers[1], stride=1)

        self.avgpool = nn.AvgPool1d(32)

        self.LSTM1 = nn.LSTM(7, hidden, 1)
        self.LSTM2 = nn.LSTM(hidden, hidden, 1)
        self.linear = nn.Linear(kernel_num * hidden, 12)
        self.softmax = nn.LogSoftmax(dim=0)

    def make_layer(self, block, out_channel, block_num, stride=1):
        block_list = []

        downsample = None
        if self.kernel_num != out_channel or stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.kernel_num, out_channel, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channel)
            )

        # Conv Block
        block_list.append(block(self.kernel_num, out_channel, stride, downsample))
        self.kernel_num = out_channel

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.kernel_num, out_channel))

        return nn.Sequential(*block_list)

    def forward(self, x):
        # print(x.shape(), x)
        out = self.conv(x)  # torch.Size= [25,32,242]
        # print('0:', out.shape)
        # out = out.reshape(25,5,32)

        out = self.Block1(out)
        # print('1:', out.shape)
        out = self.Block2(out)
        # print(out.shape)

        out = self.avgpool(out)
        # print('2:', out.shape)

        out, temp1 = self.LSTM1(out)
        # print('lstm:',out.shape)
        out, temp2 = self.LSTM2(out)
        # print(out.shape)
        out = out.view(-1, kernel_num * hidden)
        # print(out.shape)
        out = self.linear(out)
        # print(out.shape)
        out = self.softmax(out)
        # print(out.shape)
        return out


# block = Block(256, 64, stride=1, downsample=None)
# network = Net()
