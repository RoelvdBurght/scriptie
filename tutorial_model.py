# import keras as ke
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf 
import pandas as pd 

data = pd.read_csv('tutorial_data.csv', sep=',', header=None)

size_train = round(len(data) * 0.7)
size_test = len(data) - size_train

data_train = data.iloc[: size_train, :]
data_test = data.iloc[size_train: , :]

Y = data_train.iloc[:, 8]
X = data_train.iloc[:, 0:8]

Y_test = data_test.iloc[:, 8]
X_test = data_test.iloc[:, 0:8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print([Dense(2)])
# model.fit(X, Y, epochs=150, batch_size=10)

# model.evaluate(X_test, Y_test)

# print(model.metrics_names[1], scores[1]*100)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))