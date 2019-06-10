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
from keras.layers import Concatenate
# from keras.layers import Sum
from keras.layers import concatenate
# from keras.layers import Merge
import keras.optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import TensorBoard
from fbprophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose as decomp


class Model_wrapper:
	def __init__(self, filename, type_, look_back, train_split, n_out=None, residual=False):
		self.raw_data = pd.read_csv(filename)
		self.type = type_
		self.look_back = look_back
		self.n_out = n_out
		self.train_split = train_split
		self.residual = residual

	def prepare_data(self):
		'''
		Make data ready to feed into model
		Saves data in model object
		'''
		train_set, test_set = self.split_and_scale()
		# If residual connections are to be used, save the data in correct way
		if self.residual:
			self.trainX, self.trainY, self.trainX_res = self.create_dataset(train_set, self.raw_trainY.values)
			self.testX, self.testY, self.testX_res = self.create_dataset(test_set, self.raw_testY.values)

		# Else save just the regular datasets
		else:
			self.trainX, self.trainY = self.create_dataset(train_set, self.raw_trainY)
			self.testX, self.testY = self.create_dataset(test_set, self.raw_testY)

	def split_and_scale(self):
		'''
		Split and scale self.raw_data
		Returns train and test sets
		'''
		scaler = MinMaxScaler(feature_range=(0, 1))
		train_size = round(self.train_split * len(self.raw_data))
		self.train_dates = self.raw_data.loc[ : train_size, 'Date']
		self.test_dates = self.raw_data.loc[train_size : , 'Date']


		data = self.raw_data.drop('Date', axis=1)

		# Decouple target var from train data
		data_train = data.iloc[: train_size, :]
		self.raw_trainY = data_train['Target']
		# self.raw_trainY = trainY
		# data_train.drop('Target', axis=1, inplace=True)
		
		# Decouple target var from test data
		data_test = data.iloc[train_size :, :]		
		self.raw_testY = data_test['Target']
		# self.raw_testY = testY
		# data_test.drop('Target', axis=1, inplace=True)

		# ------------------------------------------------ PROBEER ANDERE SCALER
		# Scale the data
		data_train = np.array(scaler.fit_transform(data_train))
		data_test = np.array(scaler.transform(data_test))
		# Reshape target variable list so it can be concatenated tot the data
		# trainY = np.array(self.raw_trainY).reshape(len(self.raw_trainY), 1)
		# testY = np.array(self.raw_testY).reshape(len(self.raw_testY), 1)

		# # Concat/append target var to data
		# data_train = np.concatenate((trainY.reshape(len(trainY), 1), data_train), axis=1)
		# data_test = np.concatenate((testY.reshape(len(testY), 1), data_test), axis=1)

		return data_train, data_test
	
	def create_dataset(self, dataset, target_variable):
		'''
		Returns modified dataset according to self.lookback and self.n_out
		Target variable must be in first column of data
		Returns X and Y data
		'''
		y_len = 1 if self.n_out == None else n_out
		lb = self.look_back
		res_lb = self.residual

		dataX, dataY = [], []
		for i in range(len(dataset)-lb+1):
			a = dataset[i:(i+lb), :]
			# b = target_variable
			b = target_variable[i+lb:i+lb+y_len]
			if len(b)<y_len:
				break
			dataX.append(a)
			dataY.append(b)

		# ----------------------------------------------------SCHRIKKELJAREN DOEN HET NIET/WORDEN NIET MEEGENOMEN
		if self.residual:
			dataX_res, dataX, dataY = [], [], []
			# Get the same values as normal way (above) but without the first x entrys
			# where x is the length of the residual lookback

			for i in range(len(dataset) - lb + 1):
				a = dataset[i + res_lb : i + res_lb + lb, :]
				# b = dataset[i + res_lb + lb : i + res_lb + lb + n_out, 0]
				b = target_variable[i + res_lb + lb : i + res_lb + lb + n_out]
				c = dataset[i : i + lb, 0]
				if len(b)<y_len:
					break
				dataX.append(a)
				dataY.append(b)
				dataX_res.append(c)

			# Reshape dataX_res to be of form samples, timesteps, features
			dataX_res = np.array(dataX_res)
			dataX_res = dataX_res.reshape(dataX_res.shape[0], dataX_res.shape[1], 1)

			return np.array(dataX), np.array(dataY), dataX_res

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

		elif self.type is 'extended_m2m_stacked_lstm':
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			lstm1 = CuDNNLSTM(lstm_hidden, return_sequences=True, name='lstm1')(input_layer)
			lstm2 = CuDNNLSTM(lstm_hidden, return_sequences=True, name='lstm2')(lstm1)
			encoder = CuDNNLSTM(lstm_hidden, name='encoder')(lstm2)
			enc_dense1 = Dense(64, activation=activation)(encoder)
			repeat = RepeatVector(self.n_out, name='repeat')(enc_dense1)
			decoder_lstm1 = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder1')(repeat)
			decoder_lstm2 = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder2')(decoder_lstm1)
			tds = TimeDistributed(Dense(64, activation=activation))(decoder_lstm2)
			dec_dense1 = (Dense(64, activation=activation))(tds)
			dec_dense2 = (Dense(32, activation=activation))(dec_dense1)
			dense_out = Dense(1)(dec_dense2)
			out_layer = Flatten()(dense_out)

		elif self.type is 'residual_m2m':
			# Reshape trainX_res to be of shape (samples, timesteps, features)
			# self.trainX_res = self.trainX_res.reshape(self.trainX_res.shape[0], self.trainX_res.shape[1], 1)

			aux_samples, aux_timesteps, aux_features = self.trainX_res.shape
			
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			encoder = CuDNNLSTM(lstm_hidden, return_sequences=True, name='encoder')(input_layer)
			tds_enc = TimeDistributed(Dense(64, activation=activation))(encoder)

			aux_input_layer = Input(shape=(aux_timesteps, aux_features), name='aux_input')
			aux_lstm = CuDNNLSTM(lstm_hidden, return_sequences=True, name='aux_lstm')(aux_input_layer)
			aux_tds = TimeDistributed(Dense(64, activation=activation))(aux_lstm)

			merged = Concatenate(axis=1)([tds_enc, aux_tds])

			enc_dense1 = Dense(64, activation=activation, name='enc_dense1')(merged)
			repeat = RepeatVector(self.n_out, name='repeat')(Flatten()(enc_dense1))

			decoder = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder')(repeat)
			tds_out = TimeDistributed(Dense(1, activation=activation))(decoder)
			out_layer = Flatten()(tds_out)

		elif self.type is 'residual_extended_m2m':
			# Reshape trainX_res to be of shape (samples, timesteps, features)
			# self.trainX_res = self.trainX_res.reshape(self.trainX_res.shape[0], self.trainX_res.shape[1], 1)

			aux_samples, aux_timesteps, aux_features = self.trainX_res.shape
			
			input_layer = Input(shape=(timesteps, features), name='input_layer')
			encoder = CuDNNLSTM(lstm_hidden, return_sequences=True, name='encoder')(input_layer)
			tds_enc = TimeDistributed(Dense(64, activation=activation))(encoder)

			aux_input_layer = Input(shape=(aux_timesteps, aux_features), name='aux_input')
			aux_lstm = CuDNNLSTM(lstm_hidden, return_sequences=True, name='aux_lstm')(aux_input_layer)
			aux_tds = TimeDistributed(Dense(64, activation=activation))(aux_lstm)

			merged = Concatenate(axis=1)([tds_enc, aux_tds])

			enc_dense1 = Dense(64, activation=activation, name='enc_dense1')(merged)
			enc_dense2 = Dense(32, activation=activation, name='enc_dense2')(enc_dense1)
			enc_dense3 = Dense(16, activation=activation, name='enc_dense2')(enc_dense2)
			repeat = RepeatVector(self.n_out, name='repeat')(Flatten()(enc_dense3))

			decoder = CuDNNLSTM(self.n_out, return_sequences=True, name='decoder')(repeat)
			tds_dec1 = TimeDistributed(Dense(64, activation=activation))(decoder)
			tds_dec2 = TimeDistributed(Dense(32, activation=activation))(tds_dec1)
			tds_out = TimeDistributed(Dense(16, activation=activation))(tds_dec2)
			
			out_layer = Flatten()(tds_out)
	
		elif self.type is 'conv_res_m2m':
			pass

		if self.type is 'residual_m2m':
			self.model = Model([input_layer, aux_input_layer], out_layer)
		else:
			self.model = Model(input_layer, out_layer)
		plot_model(self.model, to_file='models/test_plot.png')
		print(self.model.summary())

	def train_model(self, nr_epochs, model_save_file, log_dir, opt, val_split=0.2, patience=25):

		self.model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])

		# simple early stopping and checkpoint callback
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
		mc = ModelCheckpoint(model_save_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir=log_dir)

		if self.residual:
			self.history = self.model.fit([self.trainX, self.trainX_res], self.trainY, epochs=nr_epochs, validation_split=val_split, callbacks=[es, mc, tensorboard])
		else:
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

		else:
			if not self.residual:

				results_test = self.model.predict(self.testX)
				results_train = self.model.predict(self.trainX)
				
			else:
				results_test = self.model.predict([self.testX, self.testX_res])
				results_train = self.model.predict([self.trainX, self.trainX_res])

			target_test = self.testY
			target_train = self.trainY
			x_axis_test = [x for x in range(len(target_test))]
			x_axis_train = [x for x in range(len(target_train))]
			x_plot_interval = 30

			# Train set
			fig = plt.figure(1, figsize=(10,10))
			ax1 = fig.add_subplot(211)
			ax1.set_xlabel('Date')
			ax1.set_ylabel('Send volume total')
			ax1.set_title('Train set')

			# set x ticks
			tick_locations = np.arange(min(self.raw_trainY), max(self.raw_trainY), x_plot_interval)
			x_labels = self.train_dates.iloc[::x_plot_interval]
			ax1.xaxis.set_ticks(tick_locations)
			ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
			ax1.tick_params(axis='x', labelsize='7', labelright=True)

			for idx, result in enumerate(results_train):
				if self.residual:
					axis_temp = [x for x in range(self.look_back + self.residual + idx, self.look_back + self.residual + idx + len(result))]
				else:
					axis_temp = [x for x in range(self.look_back + idx, self.look_back + idx + len(result))]
				ax1.plot(axis_temp, result, c='r')
			ax1.plot([x for x in range(len(self.raw_trainY))], self.raw_trainY, c='g')

			# Test set
			ax2 = fig.add_subplot(212)
			ax2.set_xlabel('Date')
			ax2.set_ylabel('Send volume total')
			ax2.set_title('Test set')
			
			# set x ticks
			tick_locations = np.arange(min(self.raw_testY), max(self.raw_testY), x_plot_interval)
			print(tick_locations)
			x_labels = self.test_dates.iloc[::x_plot_interval]
			ax2.xaxis.set_ticks(tick_locations)
			ax2.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
			ax2.tick_params(axis='x', labelsize='7', labelright=True)
			
			for idx, result in enumerate(results_test):
				if self.residual:
					axis_temp = [x for x in range(self.look_back + self.residual + idx, self.look_back + idx + self.residual + len(result))]
				else:
					axis_temp = [x for x in range(self.look_back + idx, self.look_back + idx  + len(result))]
				ax2.plot(axis_temp, result, c='r')
			ax2.plot([x for x in range(len(self.raw_testY))], self.raw_testY, c='g')

			plt.plot()
			plt.show()

		if show_last:
			
			# Get latest possible sample to predict from test set and plot the prediction
			target = self.testY[-1]
			input_ = self.testX[-1]
			prediction = self.model.predict(input_.reshape(1, input_.shape[0], input_.shape[1]))

			fig = plt.figure(3, figsize=(10,10))
			ax1 = fig.add_subplot(111)
			x_axis = [x for x in range(len(target))]
			ax1.plot(x_axis, target, c='g')
			ax1.plot(x_axis, prediction[0], c='r')

			plt.plot()
			plt.show()

	def explore_data(self):

		fig = plt.figure(1, figsize=(10,10))
		ax1 = fig.add_subplot(111)
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Sendvolume Total')
		ax1.plot([x for x in range(len(self.raw_data))], self.raw_data.Target)
		plt.plot()
		plt.show()

		# Preform seaonality analysis
		# m = Prophet()
		# dates = pd.to_datetime(self.raw_data['Date'])
		# send_volumes = self.raw_data['Target']
		# df = pd.DataFrame({'DS' : dates, 'Y' : send_volumes})
		# df.set_index('DS', drop=True, inplace=True)
		# print(df)
		# joe = decomp(df, freq=365)
		# trend = joe.trend
		# seasonal = joe.seasonal
		# residual = joe.resid
		# joe.plot()
		# plt.show()
		# fig2 = plt.figure(2, figsize=(12,18))
		# ax1 = fig2.add_subplot(3,1,1)
		# ax1.plot()
		# m.fit(df)


data_file = 'clean_v3_2013.csv'
mod_type = 'residual_m2m'
look_back = 45
n_out = 45
train_split = 0.7
nr_epochs = 3500
val_split = 0.2
patience = 750
learning_rate = 0.3
opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt_name = 'adam'
activation=None
res_lookback = 365

model_message = 'res_m2m'
model_id = '{}_{}_{}_lb{}_out{}_ep{}_lr{}_opt{}'.format(model_message, data_file, mod_type, look_back, n_out, nr_epochs, learning_rate, opt_name)
model_save_file = 'models/new_start/{}.h5'.format(model_id)
log_dir = 'logs/new_start/{}'.format(model_id)

test_mod = Model_wrapper(data_file, mod_type, look_back, train_split, n_out=n_out, residual=res_lookback)
test_mod.prepare_data()
test_mod.build_model(activation=activation)
test_mod.train_model(nr_epochs, model_save_file, log_dir, opt, val_split=val_split, patience=patience)
test_mod.show_results(show_last=True)
# test_mod.explore_data()