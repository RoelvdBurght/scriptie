import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import RepeatVector
import keras.optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard


class Model_wrapper:
	def __init__(self, filename, type_, look_back, train_split, n_out=None):
		self.raw_data = pd.read_csv(filename)
		self.type = type_
		self.look_back = look_back
		self.n_out = n_out
		self.train_split = train_split

	def prepare_data(self):
		'''
		Make data ready to feed into model
		Saves data in model object
		'''
		train_set, test_set = self.split_and_scale()
		self.trainX, self.trainY = self.create_dataset(train_set)
		self.testX, self.testY = self.create_dataset(test_set)

	def split_and_scale(self):
		'''
		Split and scale self.raw_data
		Returns train and test sets
		'''
		scaler = MinMaxScaler(feature_range=(0, 1))
		train_size = round(self.train_split * len(self.raw_data))

		data = self.raw_data.drop('Date', axis=1)

		# Decouple target var from train data
		data_train = data.iloc[: train_size, :]
		trainY = data_train['Target']
		self.raw_trainY = trainY
		# data_train.drop('Target', axis=1, inplace=True)
		
		# Decouple target var from test data
		data_test = data.iloc[train_size :, :]		
		testY = data_test['Target']
		self.raw_testY = testY
		# data_test.drop('Target', axis=1, inplace=True)

		# ------------------------------------------------ DATA NA SCALEN BEHOUD GEEN ONDERLINGE GROOTE
		#  ------------------------------------------------------ WAAROM IS 150 IN DATA TEST NA SCALEN EVEN GROOT ALS 100 IN DATA TRAIN
		# Scale the data
		data_train = np.array(scaler.fit_transform(data_train))
		data_test = np.array(scaler.transform(data_test))
		
		# Reshape target variable list so it can be concatenated tot the data
		trainY = np.array(trainY).reshape(len(trainY), 1)
		testY = np.array(testY).reshape(len(testY), 1)

		# Concat/append target var to data
		data_train = np.concatenate((trainY.reshape(len(trainY), 1), data_train), axis=1)
		data_test = np.concatenate((testY.reshape(len(testY), 1), data_test), axis=1)

		return data_train, data_test
	
	def create_dataset(self, dataset):
		# ------------------------------------------ hier geef je de vorige target datapunten nog niet mee als input features!!!!!!!!!!!!
		'''
		Returns modified dataset according to self.lookback and self.n_out
		Target variable must be in first column of data
		Returns X and Y data
		'''
		y_len = 1 if self.n_out == None else n_out
		lb = self.look_back

		dataX, dataY = [], []
		for i in range(len(dataset)-lb+1):
			a = dataset[i:(i+lb), 1:]
			b = dataset[i+lb:i+lb+y_len, 0]
			if len(b)<y_len:
				break
			dataX.append(a)
			dataY.append(b)
		return np.array(dataX), np.array(dataY)


	def build_model(self, lstm_hidden=64, dense_hidden=32, activation=None):
		samples, timesteps, features = self.trainX.shape

		if self.type is 'basic':
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			lstm1 = CuDNNLSTM(lstm_hidden, name='lstm_layer_1')(input_layer)
			dense1 = Dense(dense_hidden, name='dense_layer_1', activation=activation)(lstm1)
			out_layer = Dense(self.n_out, name='out_layer', activation=activation)(dense1)
		
		elif self.type is 'basic_m2m':
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			encoder = CuDNNLSTM(lstm_hidden, name='encoder')(input_layer)
			repeat = RepeatVector(self.n_out, name='repeat')(encoder)
			decoder = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder')(repeat)
			tds = TimeDistributed(Dense(64))(decoder)
			dense1 = Dense(1)(tds)
			out_layer = Flatten()(dense1)

		elif self.type is 'extended_m2m':
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			encoder = CuDNNLSTM(lstm_hidden, name='encoder')(input_layer)
			enc_dense1 = Dense(64, activation=activation)(encoder)
			repeat = RepeatVector(self.n_out, name='repeat')(enc_dense1)
			decoder = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder')(repeat)
			tds = TimeDistributed(Dense(64, activation=activation))(decoder)
			dec_dense1 = (Dense(64, activation=activation))(tds)
			dec_dense2 = (Dense(32, activation=activation))(dec_dense1)
			dense_out = Dense(1)(dec_dense2)
			out_layer = Flatten()(dense_out)

		self.model = Model(input_layer, out_layer)
		print(self.model.summary())
	
	def train_model(self, nr_epochs, model_save_file, log_dir, opt, val_split=0.2, patience=25):

		self.model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])

		# simple early stopping and checkpoint callback
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
		mc = ModelCheckpoint(model_save_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir=log_dir)

		self.history = self.model.fit(self.trainX, self.trainY, epochs=nr_epochs, validation_split=val_split, callbacks=[es, mc, tensorboard])

	def show_results(self, show_last=False):

		if self.n_out == 1:

			results_test = self.model.predict(self.testX)
			results_train = self.model.predict(self.trainX)
			target_test = self.testY
			target_train = self.trainY

			x_axis_test = [x for x in range(len(target_test))]
			x_axis_train = [x for x in range(len(target_train))]

			fig = plt.figure(1, figsize=(10,10))
			ax1 = fig.add_subplot(211)
			ax1.set_xlabel('Date')
			ax1.set_ylabel('Send volume total')
			ax1.scatter(x_axis_train, results_train, c='r')
			ax1.scatter(x_axis_train, target_train, c='g')

			ax2 = fig.add_subplot(212)
			ax1.set_xlabel('Date')
			ax1.set_ylabel('Send volume total')
			ax2.scatter(x_axis_test, results_test, c='r')
			ax2.scatter(x_axis_test, target_test, c='g')

			plt.plot()
			plt.show()
			results_test = self.model.predict(self.testX)
			results_train = self.model.predict(self.trainX)
			target_test = self.testY
			target_train = self.trainY

		elif not show_last:

			results_test = self.model.predict(self.testX)
			results_train = self.model.predict(self.trainX)
			target_test = self.testY
			target_train = self.trainY

			x_axis_test = [x for x in range(len(target_test))]
			x_axis_train = [x for x in range(len(target_train))]

			fig = plt.figure(1, figsize=(10,10))
			ax1 = fig.add_subplot(211)
			ax1.set_xlabel('Date')
			ax1.set_ylabel('Send volume total')
			ax1.set_title('Train set')
			for idx, result in enumerate(results_train):
				axis_temp = [x for x in range(self.look_back + idx, self.look_back + idx + len(result))]
				ax1.plot(axis_temp, result, c='r')
			ax1.plot([x for x in range(len(self.raw_trainY))], self.raw_trainY, c='g')

			ax2 = fig.add_subplot(212)
			ax2.set_xlabel('Date')
			ax2.set_ylabel('Send volume total')
			ax2.set_title('Test set')
			for idx, result in enumerate(results_test):
				axis_temp = [x for x in range(self.look_back + idx, self.look_back + idx + len(result))]
				ax2.plot(axis_temp, result, c='r')
			ax2.plot([x for x in range(len(self.raw_testY))], self.raw_testY, c='g')

			plt.plot()
			plt.show()
			results_test = self.model.predict(self.testX)
			results_train = self.model.predict(self.trainX)
			target_test = self.testY
			target_train = self.trainY
	
		if show_last:
			
			# Get latest possible sample to predict from test set
			target = self.testY[-1]
			print(target)
		

	def explore_data(self):

		fig = plt.figure(1, figsize=(10,10))
		ax1 = fig.add_subplot(111)
		ax1.plot([x for x in range(len(self.raw_data))], self.raw_data.Target)
		plt.plot()
		plt.show()


data_file = 'clean_v3_2013.csv' #'clean_v3_2013.csv'
mod_type = 'extended_m2m_testing_throw_this_away'
look_back = 365
n_out = 45
train_split = 0.7
nr_epochs = 10
val_split = 0.2
patience = 350
opt = keras.optimizers.Adam()
activation=None

model_id = '{}_m2m_extended_lb{}_out{}_ep{}'.format(data_file, look_back, n_out, nr_epochs)
model_save_file = 'models/new_start/{}.h5'.format(model_id)
log_dir = 'logs/new_start/{}'.format(model_id)

test_mod = Model_wrapper(data_file, mod_type, look_back, train_split, n_out=n_out)
test_mod.prepare_data()
# test_mod.build_model(activation=activation)
# test_mod.train_model(nr_epochs, model_save_file, log_dir, opt, val_split=val_split, patience=patience)
test_mod.show_results(show_last=True)
# test_mod.explore_data()