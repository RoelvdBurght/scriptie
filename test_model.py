import pandas as pd
from keras.utils import plot_model
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Input
from keras.layers import Multiply
import keras.optimizers
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# import json

# time_steps = 10
# features = 5


# _input = Input(shape=(time_steps, features))
# activations = LSTM(128, input_shape=(time_steps, features), return_sequences=True)(_input)

# # compute importance for each step
# attention = Dense(1, activation='tanh')(activations)

# # plot_model(attention, to_file='figures/model.png', show_shapes=True)

from keras.layers.core import*
from keras.models import Sequential
from keras.utils import plot_model

input_dim = 32
hidden = 32
step = 5

#The LSTM  model -  output_shape = (batch, step, hidden)
model1 = Sequential()
model1.add(LSTM(input_dim=input_dim, output_dim=hidden, input_length=step, return_sequences=True))

#The weight model  - actual output shape  = (batch, step)
# after reshape : output_shape = (batch, step,  hidden)
model2 = Sequential()
model2.add(Dense(input_dim=input_dim, output_dim=step))
model2.add(Activation('softmax')) # Learn a probability distribution over each  step.
#Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
model2.add(RepeatVector(hidden))
model2.add(Permute((2, 1)))

#The final model which gives the weighted sum:
model = Sequential()
model.add(Multiply()([model1, model2]))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
model.add(TimeDistributedMerge('sum')) # Sum the weighted elements.

plot_model(model, to_file='figures/model_testertjjjj.png', show_shapes=True)
