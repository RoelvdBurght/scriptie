import pandas as pd
from keras.utils import plot_model
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
import keras.optimizers
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json



# Globals
# np.random.seed(10)
data_set =  'clean_v2_extended.csv'
look_back =  730
train_split = 0.75
scaler = MinMaxScaler(feature_range=(0, 1))

epoch_range = [100, 250, 500, 750, 1000]


scaler = MinMaxScaler(feature_range=(0, 1))
opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
no_epochs = 10

# Functies
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	dataset = dataset.values
	for i in range(len(dataset)-look_back+1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
	return np.array(dataX)

def scale(array, scaler):
	if len(array.shape) > 1:
		return_arr = []
		for arr in array:
			return_arr.append(scaler.fit_transform(arr))
		return np.asarray(return_arr)
	else:
		joe = array.reshape(-1,1)
		joe = scaler.fit_transform(joe)
		return joe

def compute_mae(predictions, real_data):
	""" Computes the mean absolute error as wel as the error per timestep 
		Returns mean absolute error as int and error per timestep as list"""
	mae = 0
	error_axis = []
	for i, result in enumerate(predictions):
		error = abs(int(result) - int(real_data[i]))
		mae += error
		error_axis.append(error)
	mae = int(mae/(i+1))
	return mae, error_axis

def calc_error_percent(results, actual):
	""" Returns a list containing the percentage each value
		in results is to the corresponding value in actual
		Also return the average error percentage
	"""
	error_axis_percentages = []
	cumsum = 0
	for i in range(len(results)):
		joe = results[i]/actual[i]*100
		error_axis_percentages.append(joe)
		cumsum += joe
	return error_axis_percentages, int(cumsum/(i+1))


# Lees data in en sla de index en column labels op
data = pd.read_csv(data_set)
print(len(data))
index = data.loc[:, 'Date']
columns = list(data.columns)
columns.remove('Date')

# Drop de rijen waar nan in voorkomt en de kolom met datums
data.dropna(how='any', inplace=True)
data = data.drop(labels='Date', axis=1)

# Koppel de target variable los van dataframe
Y_data = data.iloc[look_back-1:, list(data.columns).index('Order volumes total')]
data = data.drop(labels='Order volumes total', axis=1)
columns.remove('Order volumes total')

# Scale de data
data = scaler.fit_transform(data)
data = pd.DataFrame(data=data, index=list(index), columns=columns)

# "Batch" de data zodat er time series analyse op gedaan kan worden
X_batch = create_dataset(data, look_back)
print(X_batch)
print(X_batch.shape)
Y_max = int(Y_data.nlargest(n=1).iloc[0])
Y_min = int(Y_data.nsmallest(n=1).iloc[0])
Y_data = Y_data.values


# Split de train en test data
size_train = round(len(data) * train_split)
size_test = len(data)-size_train
#-------------------------------------------------------------- HIER MOET JE EVENTUEEL DE DATA RANDOMIZEN --------------------------------
#-------------------------------------------------------------- WSS ANDERE SCALER WANT MINMAX HOUD VERSCHILLEN IN STAND - NIET HANDIG VOOR CATEGORISCHE DATA --------------------------------

X_train = X_batch[0 : size_train]
X_test = X_batch[size_train : ]

Y_train = Y_data[0 : size_train]
print(len(Y_train))
Y_test = Y_data[size_train : ]
Y_train_labels = index.iloc[ : size_train]
Y_test_labels = index.iloc[size_train : ]
# raise SystemExit(0)

# Create model and train
samples, time_steps, features = X_train.shape

in_layer = Input(shape=(time_steps, features))
lstm_layer = LSTM(10)(in_layer)
out_layer = Dense(1)(lstm_layer)

model = Model(inputs=in_layer, outputs=out_layer)

plot_model(model, to_file='test_model.png')