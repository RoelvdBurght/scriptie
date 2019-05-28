import pandas as pd
from keras.utils import plot_model
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.optimizers
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pprint as pp
import tensorflow as tf

# Globals
# np.random.seed(10)
data_set = 'clean_v2_extended.csv'
look_back =  730
train_split = 0.6

scaler = MinMaxScaler(feature_range=(0, 1))
opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
no_epochs = 75

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
model = Sequential()
model.add(LSTM(features, input_shape=(time_steps, features), return_sequences=True))
model.add(LSTM(features))#, return_sequences=True))
# model.add(LSTM(64))
model.add(Dense(features))
model.add(Dense(features))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
# # model.add(Dense(64))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
print(model.summary())

history = model.fit(X_train, Y_train, epochs=no_epochs, validation_split=0.2) #validation_data=(X_test, Y_test))

# Predict test and training sets
results = model.predict(X_test)
results_training = model.predict(X_train)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("loss = ", loss)
print("accuracy = ", accuracy)

# Compute mean absolute error for train and test set
mae, error_axis = compute_mae(results, Y_test)
mae_train, error_axis_train = compute_mae(results_training, Y_train)
print("Mean Absolute Error = ", mae)

# Defineer plot globals test set plot
x_plot_interval = 15
x_axis = np.linspace(1, len(results), len(results))
x_labels = Y_test_labels.iloc[::x_plot_interval]
x_locs = np.arange(min(x_axis), max(x_axis), x_plot_interval)

# Plot scatterplot met resultaten test set
fig = plt.figure(1, figsize=(10,10))
plt.figtext(0.3, 0.9, "Test set")
ax1 = fig.add_subplot(321)
ax1.scatter(x_axis, results, c='r')
ax1.scatter(x_axis, Y_test, c='g')
ax1.xaxis.set_ticks(x_locs)
ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
ax1.tick_params(axis='x', labelsize='7', labelright=True)
ax1.set_xlabel("Date")
ax1.set_ylabel("Total sales")
ax1.legend(["Predicted", 'Actual'], loc='upper right')

# Plot graph met error per voorspelling op de test set
ax2 = fig.add_subplot(323)
ax2.plot(x_axis, error_axis)
ax2.xaxis.set_ticks(x_locs)
ax2.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
ax2.axhline(y=mae, linewidth=1, c='g', linestyle='--')
ax2.tick_params(axis='x', labelsize='7', labelright=True)
ax2.set_xlabel('Date')
ax2.set_ylabel('Absolute prediction error')
plt.figtext(.02, .02, "Mean Absolute Error = {}".format(mae))

# Plot error as a percentage of sales per day
ax6 = fig.add_subplot(325)
error_perc_axis, avg_perc_error = calc_error_percent(error_axis, Y_test)
ax6.plot(x_axis, error_perc_axis)
ax6.axhline(y=avg_perc_error, linewidth=1, c='g', linestyle='--')
y_locs = np.arange(0, 100, 20)
y_locs = np.append(y_locs, avg_perc_error)
y_labels = [str(x) for x in np.arange(0, 100, 20)]
y_labels.append(str(avg_perc_error))
ax6.yaxis.set_ticks(y_locs)
ax6.yaxis.set_ticklabels(y_labels)
ax6.xaxis.set_ticks(x_locs)
ax6.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
ax6.tick_params(axis='x', labelsize='7', labelright=True)
ax6.set_xlabel('Date')
ax6.set_ylabel('Prediction error as percentage of actual sales')
ax6.set_ylim(bottom=0, top=100)


# Defineer plot globals training set plot
x_plot_interval = 15
x_axis = np.linspace(1, len(results_training), len(results_training))
x_labels = Y_train_labels.iloc[::x_plot_interval]
x_locs = np.arange(min(x_axis), max(x_axis), x_plot_interval)

# Plot scatterplot met resultaten training set
# fig = plt.figure(2, figsize=(8,8))
ax3 = fig.add_subplot(322)
plt.figtext(0.7, 0.9, "Training set")
ax3.scatter(x_axis, results_training, c='r')
ax3.scatter(x_axis, Y_train, c='g')
ax3.xaxis.set_ticks(x_locs)
ax3.xaxis.set_ticklabels(x_labels, rotation=40, ha='right')
ax3.tick_params(axis='x', labelsize='7', labelright=True)
ax3.set_xlabel("Date")
ax3.set_ylabel("Total sales")
ax3.legend(["Predicted", 'Actual'], loc='upper right')

# Plot graph met error per voorspelling op de training set
ax4 = fig.add_subplot(324)
ax4.plot(x_axis, error_axis_train)
ax4.xaxis.set_ticks(x_locs)
ax4.xaxis.set_ticklabels(x_labels, rotation=40, ha='right')
ax4.tick_params(axis='x', labelsize='7', labelright=True)
ax4.axhline(y=mae_train, linewidth=1, c='g', linestyle='--', label="MAE")
ax4.set_xlabel('Date')
ax4.set_ylabel('Absolute prediction error')
plt.figtext(0.8, .02, "Mean Absolute Error = {}".format(mae_train))

# Plot error as a percentage of sales per day
ax5 = fig.add_subplot(326)
error_perc_axis, avg_perc_error = calc_error_percent(error_axis_train, Y_train)
ax5.plot(x_axis, error_perc_axis)
ax5.axhline(y=avg_perc_error, linewidth=1, c='g', linestyle='--')
y_locs = np.arange(0, 100, 20)
y_locs = np.append(y_locs, avg_perc_error)
y_labels = [str(x) for x in np.arange(0, 100, 20)]
y_labels.append(str(avg_perc_error))
ax5.yaxis.set_ticks(y_locs)
ax5.yaxis.set_ticklabels(y_labels)
ax5.xaxis.set_ticks(x_locs)
ax5.xaxis.set_ticklabels(x_labels, rotation=40, ha='right')
ax5.tick_params(axis='x', labelsize='7', labelright=True)
ax5.set_xlabel('Date')
ax5.set_ylabel('Prediction error as percentage of actual sales')
ax5.set_ylim(bottom=0, top=100)

# Plot de training en validation loss
fig = plt.figure(2, figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.yaxis.set_label('Loss')
ax1.xaxis.set_label('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

plt.plot()
plt.show()


# plot_model(model, to_file='figures/model.png', show_shapes=True)
