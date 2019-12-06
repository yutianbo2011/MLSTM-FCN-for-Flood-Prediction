from reprocessing_prepare_input_data import reprocessing
import pandas
import networkx as nx
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import numpy as np
import heapq

def get_x_series(elevation_df, rainfall_df):
	# x: (number of samples, number of variates=2, length of series=96)
	elevation_df=elevation_df.drop(columns=['time'])
	sequence_length = 96
	number_of_samples=elevation_df.shape[1]*(elevation_df.shape[0]-sequence_length+1)
	number_of_variable=2
	
	x=np.zeros((number_of_samples, number_of_variable, sequence_length))
	print("get_x_series", x.shape)
	col=list(elevation_df.columns)
	for i in range(elevation_df.shape[1]):
		for j in range(elevation_df.shape[0]-sequence_length+1):
			ele = np.array(list(elevation_df[col[i]])[j : j + sequence_length]).reshape((sequence_length,))
			rain = np.array(list(rainfall_df[col[i]])[j :j + sequence_length]).reshape((sequence_length,))
			x[i*(elevation_df.shape[0]-sequence_length+1)+j][0][:] = ele
			x[i*(elevation_df.shape[0]-sequence_length+1)+j][1][:] = rain
	#np.save(file_name, x)
	#print(file_name, x)
	return x

def get_y_series(elevation_df):
	elevation_df = elevation_df.drop(columns=['time'])
	sequence_length = 96
	number_of_samples=elevation_df.shape[1]*(elevation_df.shape[0]-sequence_length+1)
	y = np.zeros((number_of_samples,1 ))
	print("get_y_series", y.shape)
	col = list(elevation_df.columns)
	for i in range(elevation_df.shape[1]):
		for j in range(elevation_df.shape[0]-sequence_length+1):
			ele = list(elevation_df[col[i]])[j+sequence_length-1]
			y[i*(elevation_df.shape[0]-sequence_length+1)+j] = ele
		
	#np.save(file_name, y)
	#print(file_name, y)
	return y

def classify_y(array):
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if(array[i][j]>0): array[i][j]=0
			else : array[i][j]=1
	return array

file_root=r'D:\schoolwork\data\RA\machine_learning'
data_file_root_1708=r'D:\schoolwork\data\RA\machine_learning\2017_08'
data_file_root_1605=r'D:\schoolwork\data\RA\machine_learning\2016_05'
data_file_root_1604=r'D:\schoolwork\data\RA\machine_learning\2016_04'
channel_network_file=r'D:\schoolwork\data\RA\machine_learning\Harris_county_drainage_modified_final.csv'
channel_structure_file=r'D:\schoolwork\data\RA\machine_learning\Harris_County_channel.graphml'
sensor_match_file=file_root+'\\'+ r'Harris_County_sensor_match.xlsx'

ele_df_file_1708=data_file_root_1708+"\\"+r"output\elevation.xlsx"
ele_x_df_file_1708=data_file_root_1708+"\\"+r"output\elevation_x.xlsx"
ele_y_df_file_1708=data_file_root_1708+"\\"+r"output\elevation_y.xlsx"
rain_df_file_1708=data_file_root_1708+"\\"+r"output\rainfall.xlsx"

ele_df_file_1605=data_file_root_1605+"\\"+r"output\elevation.xlsx"
ele_x_df_file_1605=data_file_root_1605+"\\"+r"output\elevation_x.xlsx"
ele_y_df_file_1605=data_file_root_1605+"\\"+r"output\elevation_y.xlsx"
rain_df_file_1605=data_file_root_1605+"\\"+r"output\rainfall.xlsx"

ele_df_file_1604=data_file_root_1604+"\\"+r"output\elevation.xlsx"
ele_x_df_file_1604=data_file_root_1604+"\\"+r"output\elevation_x.xlsx"
ele_y_df_file_1604=data_file_root_1604+"\\"+r"output\elevation_y.xlsx"
rain_df_file_1604=data_file_root_1604+"\\"+r"output\rainfall.xlsx"

x_train_file=file_root+ "\\" + r'data_reprocess\X_train.npy'
y_train_file=file_root+ "\\" + r'data_reprocess\Y_train.npy'
x_test_file=file_root+ "\\" + r'data_reprocess\X_test.npy'
y_test_file=file_root+ "\\" + r'data_reprocess\Y_test.npy'

# time_start_1708=datetime(2017, 8, 24)
# time_end_1708=datetime(2017, 8, 31)
# time_start_1604=datetime(2016, 4, 17)
# time_end_1604=datetime(2016, 4, 24)
# time_start_1605=datetime(2016, 5, 25)
# time_end_1605=datetime(2016,6, 1)

time_start_1708=datetime(2017, 8, 26)
time_end_1708=datetime(2017, 8, 29,15)
time_start_1604=datetime(2016, 4, 17)
time_end_1604=datetime(2016, 4, 20)
time_start_1605=datetime(2016, 5, 26)
time_end_1605=datetime(2016,5, 30)


channel_model=nx.read_graphml(channel_structure_file)
network=pandas.read_csv(channel_network_file)
sensor_match=pandas.read_excel(sensor_match_file)


repro_1708=reprocessing(network, sensor_match, networkx_graph=channel_model,
								  time_start=time_start_1708, time_end=time_end_1708, file_root=data_file_root_1708)
# repro_1708.get_position_dic()
# repro_1708.get_map_size()
repro_1708.match_sensor_with_node_by_input()

elevation_dataframe_1708=repro_1708.compute_elevation_dataframe(ele_df_file_1708,name=' Stream 2017-30-08.xls' )
rainfall_dataframe_1708=repro_1708.compute_rainfall_dataframe(rain_df_file_1708, name=' Rainfall 2017-30-08.xls')
elevation_x_dataframe_1708=repro_1708.get_elevation_x(elevation_dataframe_1708, file_name=ele_x_df_file_1708)
elevation_y_dataframe_1708=repro_1708.get_elevation_y(elevation_dataframe_1708, file_name=ele_y_df_file_1708)


repro_1605=reprocessing(network, sensor_match, networkx_graph=channel_model,
								  time_start=time_start_1605, time_end=time_end_1605, file_root=data_file_root_1605)
# repro_1605.get_position_dic()
# repro_1605.get_map_size()
repro_1605.match_sensor_with_node_by_input()

elevation_dataframe_1605=repro_1605.compute_elevation_dataframe(ele_df_file_1605,name=' Stream 2016-31-05.xls' )
rainfall_dataframe_1605=repro_1605.compute_rainfall_dataframe(rain_df_file_1605, name=' Rainfall 2016-31-05.xls')
elevation_x_dataframe_1605=repro_1605.get_elevation_x(elevation_dataframe_1605, file_name=ele_x_df_file_1605)
elevation_y_dataframe_1605=repro_1605.get_elevation_y(elevation_dataframe_1605, file_name=ele_y_df_file_1605)


repro_1604=reprocessing(network, sensor_match, networkx_graph=channel_model,
								  time_start=time_start_1604, time_end=time_end_1604, file_root=data_file_root_1604)
# repro_1604.get_position_dic()
# repro_1604.get_map_size()
repro_1604.match_sensor_with_node_by_input()

elevation_dataframe_1604=repro_1604.compute_elevation_dataframe(ele_df_file_1604, name=' Stream 2016-23-04.xls')
rainfall_dataframe_1604=repro_1604.compute_rainfall_dataframe(rain_df_file_1604,name=' Rainfall 2016-23-04.xls')
elevation_x_dataframe_1604=repro_1604.get_elevation_x(elevation_dataframe_1604, file_name=ele_x_df_file_1604)
elevation_y_dataframe_1604=repro_1604.get_elevation_y(elevation_dataframe_1604, file_name=ele_y_df_file_1604)

'''

'''
def split_dataset(dataset, choice_list):
	mask = np.ones(dataset.shape[0], dtype=bool)
	mask[choice_list,] = False
	test, train = dataset[choice_list], dataset[mask]
	return test, train


X_1708=get_x_series(elevation_x_dataframe_1708, rainfall_dataframe_1708)
X_1605=get_x_series(elevation_x_dataframe_1605, rainfall_dataframe_1605)
X_1604=get_x_series(elevation_x_dataframe_1604, rainfall_dataframe_1604)

Y_1708=classify_y(get_y_series(elevation_y_dataframe_1708))
Y_1605=classify_y(get_y_series(elevation_y_dataframe_1605))
Y_1604=classify_y(get_y_series(elevation_y_dataframe_1604))

X=np.concatenate((X_1708, X_1605, X_1604), axis=0)
Y=np.concatenate((Y_1708, Y_1605, Y_1604), axis=0)

number_sample=X.shape[0]
choice_list=list(np.random.choice(number_sample, int(number_sample*0.2)))
x_test, x_train=split_dataset(X, choice_list)
y_test, y_train=split_dataset(Y, choice_list)

np.save(x_train_file, x_train)
np.save(x_test_file, x_test)
np.save(y_train_file, y_train)
np.save(y_test_file, y_test)
