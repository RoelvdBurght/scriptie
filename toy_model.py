import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import datetime


# Globals
np.random.seed(10)
data_set = 'toy_data.csv'
look_back =  3
train_split = 0.7
patience = 250

scaler = MinMaxScaler(feature_range=(0, 1))
opt = keras.optimizers.Adam()
no_epochs = 1000

# now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
# model_name = '2lstm + tdist + 3 dense' 
# model_save_file = 'models/best_{}_{}_lb_{}_spl_{}_pt_{}_{}.h5'.format(model_name, data_set[:-4], look_back, train_split, patience, now)
# tensorboard = TensorBoard(log_dir='logs/{}_{}_lb_{}_spl_{}_pt_{}_{}'.format(model_name, data_set[:-4], look_back, train_split, patience, now))

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
# data = scaler.fit_transform(data)
# data = pd.DataFrame(data=data, index=list(index), columns=columns)

# "Batch" de data zodat er time series analyse op gedaan kan worden
X_batch = create_dataset(data, look_back)
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
Y_test = Y_data[size_train : ]
Y_train_labels = index.iloc[ : size_train]
Y_test_labels = index.iloc[size_train : ]


samples, time_steps, features = X_train.shape

print(X_train)
in_test = Input(shape=(time_steps, features))


input_layer = Input(shape=(time_steps,features), name='input_layer')
lstm_layer_1 = CuDNNLSTM(time_steps, name='lstm_layer_1')(input_layer)#, return_sequences=True)(input_layer)
# flat = Flatten()(lstm_layer_1)
dense_1 = Dense(1)(lstm_layer_1)

model = Model(inputs=input_layer, outputs=dense_1)
# flat = Flatten()(lstm_layer_1)
# conv = Conv1D(1,1)(lstm_layer_1)
# lstm_layer_2 = CuDNNLSTM(features, name='lstm_layer_2', return_sequences=True)(lstm_layer_1)
# time_dist = TimeDistributed(Dense(features, name='time_dist_layer'))(lstm_layer_2)
# flat = Flatten(name='flaten')(time_dist)
# conv1 = Conv2D(1, 1, strides=1, dilation_rate=365, padding='same', name='conv_layer_1')(flat)

# dense_1 = Dense(128, name='dense_layer_1')(flat)
# # flat = Flatten()(dense_1)
# # drop_1 = Dropout(0.2)(dense_1)
# dense_2 = Dense(64, name='dense_layer_2')(dense_1)
# drop_2 = Dropout(0.2)(dense_2)
# dense_3 = Dense(32, name='dense_layer_3')(conv1)

# # time_dist = 
# out_layer = Dense(1, name='out_layer')(dense_3)

# conv_layer = Conv1D(1, 2, strides=2)(lstm_layer)
# out_layer = Dense(1, name='out_layer')(lstm_layer)

print(model.summary())

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])

# simple early stoppings
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
# mc = ModelCheckpoint(model_save_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# history = model.fit(X_train, Y_train, epochs=no_epochs, validation_split=0.2)#, callbacks=[es, mc, tensorboard]) #validation_data=(X_test, Y_test))
# print(model.layers[1].output)

# Predict test and training sets
# results = model.predict(X_test)
# results_training = model.predict(X_train)
