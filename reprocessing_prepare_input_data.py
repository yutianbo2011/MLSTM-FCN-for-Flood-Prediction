import pandas
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import xlrd
from datetime import datetime, timedelta
from xlrd import xldate_as_tuple
import math
from math import radians, cos, sin, asin, sqrt
import time
import json
import urllib.request



class reprocessing():
	def __init__(self, network, sensor_match, networkx_graph, time_start, time_end, file_root):
		self.network=network
		self.sensor_match=sensor_match
		self.nx_graph=networkx_graph
		self.file_root=file_root
		self.time_start=time_start
		self.time_end=time_end
		self.time_interval=30
		self.hours_diff=6

		self.match_sensor_with_node = {} #key: sensor, node: channel component
		self.node_ele_dic={}


	#computation of distance
	def haversine(self, lon1, lat1, lon2, lat2):
		"""
		Calculate the great circle distance between two points on the earth (specified in decimal degrees)in terms of KM
		"""
		lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
		c = 2 * asin(sqrt(a))
		r = 6371
		return c * r

	def get_map_size(self):
		east = max(self.position_all.items(),key = lambda d:d[1][0])
		west = min(self.position_all.items(),key = lambda d:d[1][0])
		north = max(self.position_all.items(),key = lambda d:d[1][1])
		south = min(self.position_all.items(),key = lambda d:d[1][1])
		print(east[1][0], west[1][0], north[1][1], south[1][1], type(east))
		#shape = (abs(east[1][0] - west[1][0]), abs(north[1][1] - south[1][1]))
		shape=(east[1][0], west[1][0], north[1][1], south[1][1])
		print("map shape ", shape, type(shape))
		self.shape=shape
		return shape

	def compute_distance(self, lon1, lat1, lon2, lat2):
		dis = math.sqrt(abs(lon1 - lon2) ** 2 + abs(lat1 - lat2) ** 2)
		return dis

	#match sensor, nodes and position
	def match_sensor_with_node_by_input(self):
		for i in range(self.sensor_match.shape[0]):
			self.match_sensor_with_node[str(self.sensor_match['file_name'][i])]=str(self.sensor_match['node'][i])
		print("match_sensor_with_node %s, %s" %(len(self.match_sensor_with_node), self.match_sensor_with_node))
		return self.match_sensor_with_node

	def get_position_dic(self):
		self.position_all={}
		for i in range(self.network.shape[0]):
			try:
				if (str(self.network.F_NODE[i]) not in self.position_all.keys()):
					self.position_all[str(self.network.F_NODE[i])] = (float(self.network.ox[i]), float(self.network.oy[i]))
				if (str(self.network.T_NODE[i]) not in self.position_all.keys()):
					self.position_all[str(self.network.T_NODE[i])] = (float(self.network.dx[i]), float(self.network.dy[i]))
			except:
				print("exception for row %s"%i)
				continue
			if(str(self.network.F_NODE[i])=='0000000000000000' or str(self.network.T_NODE[i])=='0000000000000000'):
				print("find 0000000000000000 %s" %i)
		print("number of nodes %s" % len(self.position_all.keys()), "position all", self.position_all)
		#self.shape=self.get_map_size()
		return self.position_all

	def script_google_elevation(self, lat, lng):
		apikey = "AIzaSyAXaHwrMU9EihBqmE9oNH31VX9POnovbms"
		url = "https://maps.googleapis.com/maps/api/elevation/json" + "?locations=" + str(lat) + "," + str(
			lng) + "&key=" + apikey
		request = urllib.request.urlopen(url)
		try:
			results = json.load(request).get('results')
			if 0 < len(results):
				elevation = results[0].get('elevation')
				# ELEVATION
				return elevation
			else:
				print('HTTP GET Request failed.')
		except:
			print('JSON decode failed: ' + str(request))
			raise TimeoutError

	def get_model_elevation(self, model):
		for node in model.nodes():
			ele=self.script_google_elevation(self.position_all[node][1], self.position_all[node][0])
			model.nodes[node]['elevation']=ele
		return model

	#compute elevation and rainfall dataframe
	def compute_elevation_dataframe(self,output_file, name):
		data=[]
		time = self.time_start
		while (time != self.time_end):
			t = str(time)
			t = t.replace('-', '')
			t = t.replace(' ', '')
			t = t.replace(':', '')
			t = t[:-2]
			data.append(t)
			time += timedelta(minutes=30)
		# data.append("top_of_spillway")
		number_rows=len(data)
		data=np.array(data).reshape(number_rows, 1)
		col_list=["time"]
		for i in range(self.sensor_match.shape[0]):
			threshold = float(self.sensor_match['Top_of_Spillway'][i])
			file_name = str(self.sensor_match['file_name'][i])
			col_list.append(str(self.match_sensor_with_node[file_name]))
			input_file = self.file_root + str(r"\elevation") + "\\" + file_name + name
			dis_arr = self.get_elevation_dis(input_file, threshold, self.time_start, self.time_end)
			#name = self.match_sensor_with_node[self.sensor_match['file_name'][i]] + '0'
			data=np.hstack((data, dis_arr))
		elevation_dataframe=pandas.DataFrame(data, columns=col_list)
		print("elevation_dataframe", elevation_dataframe.head())
		elevation_dataframe.to_excel(output_file, index=False)
		return elevation_dataframe

	def get_elevation_dis(self, input_file_name, threshold, start_time, end_time):
		sheetname = 'Sheet'
		data = xlrd.open_workbook(input_file_name)
		sheet = data.sheet_by_name(sheetname)
		rows = sheet.nrows
		ele_list = []
		# category_list=[-0.03, 0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3,  0.5, 1.0, 5.0, 10.0]
		time_match = self.time_start
		while (time_match != self.time_end):

			# print("start/end time ", start_time, end_time)
			# print("time match", time_match)
			ele = 0
			time = sheet.cell_value(1, 1)
			time = datetime(*xldate_as_tuple(time, 0))
			if(time<time_match):
				ele_list.append(ele_list[len(ele_list)-1])
				time_match += timedelta(minutes=30)
				continue
			for j in range(1, rows - 1):
				time = sheet.cell_value(j, 1)
				time = datetime(*xldate_as_tuple(time, 0))
				#time = int((time - self.time_start).seconds / 60 + (time - self.time_start).days * 24 * 60)
				time_former = sheet.cell_value(j + 1, 1)
				time_former = datetime(*xldate_as_tuple(time_former, 0))
				#time_former = int((time_former - self.time_start).seconds / 60 + (time_former - self.time_start).days * 24 * 60)
				elevation = sheet.cell_value(j, 2)
				elevation_former = sheet.cell_value(j, 2)
				# print("time:", time_match, time, time_former)
				if (time >= time_match and time_former < time_match):
					# print("time:", time_match, time, time_former)
					ele = elevation_former + (elevation - elevation_former) * (time_match - time_former) / (
								time - time_former)
					break
			ele_list.append(float(threshold-ele))
			time_match += timedelta(minutes=30)
		# ele_list.append(float(threshold))
		ele_list = np.array(ele_list).reshape((len(ele_list), 1))
		return ele_list

	def compute_rainfall_dataframe(self, output_file, name):
		rainfall_dataframe = pandas.DataFrame(columns=[])
		print("match_sensor_with_node", self.match_sensor_with_node)
		for i in range(self.sensor_match.shape[0]):
			file_name = str(self.sensor_match['file_name'][i])
			input_file = self.file_root + r"\rainfall" + r'\\' + file_name + name
			rain_fall_list = self.get_rainfall_list(input_file)
			#name = self.match_sensor_with_node[str(self.sensor_match['file_name'][i])] + '00'  # * 100
			# rain_dataframe = pandas.DataFrame(rain_fall_list, columns=[long(name)])
			rain_dataframe = pandas.DataFrame(rain_fall_list, columns=[str(self.match_sensor_with_node[str(self.sensor_match['file_name'][i])])])
			rainfall_dataframe = pandas.concat([rainfall_dataframe, rain_dataframe], axis=1)
		rainfall_dataframe.to_excel(output_file, index=False)
		return rainfall_dataframe

	def get_rainfall_list(self, file_name):
		sheetname = 'Sheet'
		data = xlrd.open_workbook(file_name)
		sheet = data.sheet_by_name(sheetname)
		rows = sheet.nrows
		time_star = sheet.cell_value(rows - 1, 0)
		time_star = datetime(*xldate_as_tuple(time_star, 0))
		diff_star = int((time_star - self.time_start).seconds / 3600 + (time_star - self.time_start).days * 24)  # in hours

		time_end = sheet.cell_value(1, 0)
		time_end = datetime(*xldate_as_tuple(time_end, 0))
		diff_end = int((time_end - self.time_end).seconds / 3600 + (time_end - self.time_end).days * 24)

		rainfall = []
		hours = self.hours_diff
		#rows = rows - hours  # use the sum of rainfall in 12 hours as input
		print("diff", time_star, diff_star, time_end, diff_end, rows)
		range_start=int(-diff_star+1)
		range_end=int(rows - diff_end-1-self.hours_diff)
		print(file_name,"range ",  range_start, range_end, sheet.cell_value(rows-range_start, 0), sheet.cell_value(rows-range_end, 0))
		# print("time start", datetime(*xldate_as_tuple(sheet.cell_value(rows-range_start, 0), 0)),
		#       "time end", datetime(*xldate_as_tuple(sheet.cell_value(rows-range_end, 0), 0)))
		for j in range(range_start, range_end):
			# make it the same order with natrual time
			k = rows - j
			rain = 0.0
			for h in range(hours):
				rain += float(sheet.cell_value(k - h, 2))
			rainfall.append(rain)
			rainfall.append(rain)  # every half hour, so append twice
		rainfall = np.array(rainfall).reshape((len(rainfall), 1))
		return rainfall

	def get_start_end_rain_time(self, dataframe, threshold_start, threshold_end):
		# compute the average rainfall and then judge when it stated to rain
		data = dataframe.values
		average_list = []
		rain_start_time_list = []
		rain_end_time_list = []
		row_start = []
		row_end = []
		for row in data:
			average = np.array(row).sum() / float(data.shape[1])
			average_list.append(average)
		average_list = average_list[3:] #get rid of statistics of first 3 rows
		print("average rainfall", average_list)
		rain_now = False
		#for i in range(len(average_list), 0, -1):
		for i in range(len(average_list)):
			if (average_list[i] >= threshold_start and not rain_now):
				rain_start_time_list.append(self.time_start + timedelta(minutes=(i * 60)))
				row_start.append(i)
				rain_now = True
			elif (average_list[i] < threshold_end and rain_now):
				rain_end_time_list.append(self.time_start + timedelta(minutes=(i * 60)))
				row_end.append(i)
				rain_now = False
		rainfall_ave_list_part = average_list[row_start[0]:row_end[len(row_end) - 1]]
		rainfall_average = np.mean(rainfall_ave_list_part)
		print("rain_start_time_list %s, rain_end_time_list %s, rainfall_average %s"
			  %(rain_start_time_list, rain_end_time_list, rainfall_average))
		return rain_start_time_list, rain_end_time_list, rainfall_average
	
	def get_elevation_x(self, df, file_name):
		df[:-self.hours_diff*2].to_excel(file_name, index=False)
		return df[:-self.hours_diff*2]
		
	def get_elevation_y(self, df, file_name):
		df[self.hours_diff*2 :].to_excel(file_name, index=False)
		return df[self.hours_diff*2:]



