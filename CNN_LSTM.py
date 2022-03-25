import torch
from torch import nn
from torch.nn import Conv1d, SELU, MaxPool1d, Dropout, LSTM, Linear, Softmax, Sequential


# 神经网路的搭建
class cnn_lstm(nn.Module):
    def __init__(self):
        super(cnn_lstm, self).__init__()
        self.module = Sequential(
            # nn.BatchNorm1d(num_features=1),
            Conv1d(in_channels=64, out_channels=32, kernel_size=30, stride=1),
            SELU(),
            MaxPool1d(kernel_size=4),
            Dropout(p=0.1),

            Conv1d(in_channels=32, out_channels=32, kernel_size=50, stride=1),
            SELU(),
            MaxPool1d(kernel_size=4),
            Dropout(p=0.1),

            LSTM(input_size=32, hidden_size=128, num_layers=2),
            Linear(in_features=128, out_features=12),
            # Softmax(dim=1)
        )

    def forward(self, x):
        x = self.module(x)
        return x


module = cnn_lstm()
print(module)
