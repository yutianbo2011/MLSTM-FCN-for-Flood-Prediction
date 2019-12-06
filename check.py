import pandas as pd
import numpy as np
#import keras

df=pd.DataFrame(np.random.random((8, 4)), columns=['A', 'B', 'C', 'D'])
print(df.shape)
x=np.zeros((3,3,3))
y=np.zeros((3,3,3))
z=np.concatenate((x, y), axis=0)
print(z.shape)

file_root=r'D:\schoolwork\data\RA\machine_learning'
x_train_file=file_root+ "\\" + r'data_reprocess\X_train.npy'
y_train_file=file_root+ "\\" + r'data_reprocess\Y_train.npy'
x_test_file=file_root+ "\\" + r'data_reprocess\X_test.npy'
y_test_file=file_root+ "\\" + r'data_reprocess\Y_test.npy'

x_train=np.load(x_train_file)
y_train=np.load(y_train_file)
x_test=np.load(x_test_file)
y_test=np.load(y_test_file)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("x_train", x_train[1:20], x_train[-20:])
print("x_test", x_test[1:20], x_test[-20:])
print("y_train", y_train[:20], y_train[-20:])
print("y_test", y_test[:20], y_test[-20:], y_test)
count_one=0
for i in range(y_train.shape[0]):
	if(y_train[i][0]==1):count_one+=1
print(count_one)
count_zero=0
for i in range(x_train.shape[0]):
	for j in range(x_train.shape[1]):
		for k in range(x_train.shape[2]):
			if(x_train[i][j][k]==0):
				# print(i, j, k)
				count_zero+=1
print(count_zero)