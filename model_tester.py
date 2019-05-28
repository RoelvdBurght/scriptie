import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model as lm
import matplotlib.pyplot as plt

class Plotter:
	def __init__(self, data_size, data_set, look_back, scale=False, train_split=None):
		self.data_size = data_size
		self.data_set = data_set
		self.look_back = look_back
		self.train_split = train_split

		# Lees data in en sla de index en column labels op
		self.data = self.read_data()
		self.index = self.data.loc[:, 'Date']
		columns = list(self.data.columns)
		columns.remove('Date')
		self.columns = columns

		# Drop de rijen waar nan in voorkomt en de kolom met datums
		self.data.dropna(how='any', inplace=True)
		self.data = self.data.drop(labels='Date', axis=1)

		# Scale de data (voor model tests)
		if scale:
			scaler = MinMaxScaler(feature_range=(0, 1))
			data = scaler.fit_transform(self.data)
			self.data = pd.DataFrame(data=data, index=list(self.index), columns=self.columns)

		# Koppel de target variable los van dataframe
		# self.Y_data = self.data.iloc[self.look_back-1:, list(self.data.columns).index('Order volumes total')]
		self.Y_data = self.data['Order volumes total']
		self.data = self.data.drop(labels='Order volumes total', axis=1)
		self.columns.remove('Order volumes total')

	def split_data(self, split):
		split = int(split*len(self.data))
		self.X_train = self.data.iloc[: split, :]
		self.X_test = self.data.iloc[split :, :]
		self.Y_train = self.Y_data.iloc[: split]
		self.Y_test = self.Y_data.iloc[split :]

	def read_data(self):
		if self.data_size == 'l':
			if self.data_set == 'no':
				data = pd.read_csv('clean_v2_extended_no_outliers.csv')
			elif self.data_set == 'reg':
				data = pd.read_csv('clean_v2_extended.csv')
		elif self.data_size == 's':
			if self.data_set == 'no':
				data = pd.read_csv('clean_v2_no_outliers.csv')
			elif self.data_set == 'reg':
				data = pd.read_csv('clean_v2.csv')
		if self.data_set == 'toy':
			print('joi')
			data = pd.read_csv('toy_data.csv')
		try:
			return data
		except:
			raise NameError('Data_size = s of l, Data_set = no of reg')

	def load_model(self, model_name):
		self.model = lm(model_name)
	
	def create_dataset(self):
		dataset = self.data
		dataX, dataY = [], []
		dataset = dataset.values
		for i in range(len(dataset)-self.look_back+1):
			a = dataset[i:(i+self.look_back), :]
			dataX.append(a)
		return np.array(dataX)
	
	def prep_data_for_predictions(self):

		# "Batch" de data zodat er time series analyse op gedaan kan worden
		X_batch = self.create_dataset()
		# Y_max = int(Y_data.nlargest(n=1).iloc[0])
		# Y_min = int(Y_data.nsmallest(n=1).iloc[0])
		Y_data = self.Y_data.values

		# Split de train en test data
		size_train = round(len(self.data) * self.train_split)
		size_test = len(self.data)-size_train

		self.X_train = X_batch[0 : size_train]
		self.X_test = X_batch[size_train : ]
		print(self.X_train)

		self.Y_train = Y_data[ : size_train]
		self.Y_test = Y_data[size_train : ]
		print(self.Y_train)
		print('----------', self.Y_test[look_back:].shape, '---------------------')
		self.Y_train_labels = self.index.iloc[ : size_train]
		self.Y_test_labels = self.index.iloc[size_train : ]

	def predict_test_set(self):
		self.prep_data_for_predictions()

		self.results_test = self.model.predict(self.X_test)
		print('----------', len(self.results_test), '---------------------')
		self.results_train = self.model.predict(self.X_train)

	def show_predicitons(self):        
		# Defineer plot globals test set plot
		x_plot_interval = 15
		x_axis = np.linspace(1, len(self.results_test), len(self.results_test))
		x_labels = self.Y_test_labels.iloc[::x_plot_interval]
		x_locs = np.arange(min(x_axis), max(x_axis), x_plot_interval)

		self.fig = plt.figure(1, figsize=(10,10))
		ax1 = self.fig.add_subplot(121)
		ax1.scatter(x_axis, self.results_test, c='r')
		ax1.scatter(x_axis, self.Y_test[look_back:], c='g')
		plt.show()

data_size = 'l'
data_set = 'toy'
look_back = 2
scale = False
train_split = 0.7
model_name = 'best_two layer lstm + 3 dense_clean_v2_extended_lb_14_pt_250.h5'
model_path = 'models/' + model_name

plotter = Plotter(data_size, data_set, look_back, scale, train_split)
plotter.prep_data_for_predictions()
# plotter.load_model(model_path)
# plotter.predict_test_set()
# plotter.show_predicitons()