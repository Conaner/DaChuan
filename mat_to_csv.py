# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['cmr10']

Points = sio.loadmat('data/PPG/Training_data/DATA_12_TYPE02.mat')

Array_M = Points.get('sig')

Array_1 = Array_M[1, :]

List_1 = Array_1.tolist()

# print(List_1)


plt.figure('Draw')
plt.plot(List_1)
plt.title("DATA_12_channel1")
plt.draw()

# plt.pause(1)
plt.savefig("DATA_12_channel1.jpg")

plt.close()
