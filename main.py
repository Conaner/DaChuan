import numpy as np
import scipy.io as sio

p = 0.75

dict = sio.loadmat('data/DATA_01_TYPE01.mat')

Array_sig = dict.get('sig')

Array_sig_1 = Array_sig[1, :]
Array_sig_2 = Array_sig[2, :]

List__sig_1 = Array_sig_1.tolist()
List__sig_2 = Array_sig_2.tolist()

length1 = len(List__sig_1)
length2 = len(List__sig_2)

List_train1 = List__sig_1[:p*length1]
List_test1 = List__sig_1[p*length1:]

List_train2 = List__sig_1[:p*length2]
List_test2 = List__sig_1[p*length2]



# print(List__sig_1)
# print(List__sig_2)
