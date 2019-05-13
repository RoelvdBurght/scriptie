import pandas as pd
from keras.utils import plot_model
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pprint as pp

# Globals
np.random.seed(10)

look_back = 7
train_split = 0.7

scaler = MinMaxScaler(feature_range=(0, 1))

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


# Lees data in en sla de index en column labels op
data = pd.read_csv('clean_v2.csv')
data.dropna(how='any', inplace=True) # Drop alle rijen waar Nan in voorkomt
index = data.loc[look_back-1:, 'Date']
columns = data.columns
data = data.drop(labels='Date', axis=1)

# Verander de datatypes van de kolommen naar float
for index, string in data['rain_time_in_min'].iteritems():
	data.loc[index, 'rain_time_in_min'] = int(string.replace(',', ''))

# "Batch" de data zodat er time series analyse op gedaan kan worden
# Voeg index labels toe aan Y data
X_batch = create_dataset(data, look_back)
Y_data = data.loc[look_back-1:, 'Order volumes total']
Y_max = int(Y_data.nlargest(n=1).iloc[0])
Y_min = int(Y_data.nsmallest(n=1).iloc[0])
Y_data = Y_data.values


# Split de train en test data
size_train = round(len(data) * train_split)
size_test = len(data)-size_train
#-------------------------------------------------------------- JE MOET HIER AL JE DATA SCALEN, ANDERS GAAT HET FOUT BIJ HET MINMAX OMDAT DIE 1 GELIJK ZET AAN DE
#-------------------------------------------------------------- DE HOOGSTE WAARDE IN JE DATASET EN DIE DUS VERSCHILLEN ALS JE ZE EERST SPLIT
#-------------------------------------------------------------- HIER MOET JE EVENTUEEL DE DATA RANDOMIZEN --------------------------------
#-------------------------------------------------------------- WSS ANDERE SCALER WANT MINMAX HOUD VERSCHILLEN IN STAND - NIET HANDIG VOOR CATEGORISCHE DATA --------------------------------

X_train = X_batch[0 : size_train]
X_test = X_batch[size_train : ]

Y_train = Y_data[0 : size_train]
Y_test = Y_data[size_train : ]

# ------------------------------------------------------ DEPRECATED
# data_train = data.iloc[0:size_train, :]
# data_test = data.iloc[size_train:, :]

# Y_train = data_train.loc[:,'Grand Total']
# Y_test = data_test.loc[:,'Grand Total']

# X_train = data_train.drop(columns=['Grand Total'])
# X_test = data_test.drop(columns=['Grand Total'])
# ------------------------------------------------------ DEPRECATED


# Scale the feature variables between 0 and 1
X_train = scale(X_train, scaler)
X_test = scale(X_test, scaler)

Y_train = scale(Y_train, scaler)
Y_test = scale(Y_test, scaler)
samples, time_steps, features = X_train.shape
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Create model and train
model = Sequential()
# model.add(LSTM(32, batch_input_shape=(samples, time_steps, features)))
model.add(LSTM(64, input_shape=(time_steps, features)))
# model.add(Dense(42))
# model.add(LSTM(32))
model.add(Dense(64))
model.add(Dense(1))

# print(len(X_train))
# print(X_train.shape)
# print(Y_train)
# sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
print(model.summary())

history = model.fit(X_train, Y_train, epochs=1000, validation_data=(X_test, Y_test))

results = model.predict(X_test)

print(len(results))
print(len(Y_test))

x_axis = np.linspace(1, len(results), len(results))
print(len(x_axis))
plt.scatter(x_axis, results, c='r')
plt.scatter(x_axis, Y_test, c='g')
plt.plot()
plt.show()
# model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)
# joe = np.array([  [[1,1], [1,1], [1,1]],
#                     [[2,2],[2,2],[2,2]],
#                     [[3,3],[3,3],[3,3]]])
# print(joe.shape)
# print(X_train.shape)
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)


# print(len(results))
# x_axis = np.linspace(1, 134,  133)
# print(len(y_axis))
# plt.scatter(x_axis, results, c='r')
# plt.scatter(x_axis, Y_test, c='g')
# plt.plot()
# plt.show()


# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# plot_model(model, to_file='model.png', show_shapes=True)
