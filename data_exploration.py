import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Globals
x_plot_interval = 20
look_back = 7

# Lees data in en sla de index en column labels op
data = pd.read_csv('clean_v2.csv')
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

fig = plt.figure(1, figsize=(8,8))
ax1 = fig.add_subplot(111)
x_axis = np.arange(0, len(Y_data), 1)

ax1.plot(x_axis, Y_data)
ax1.xaxis.set_ticks(x_axis[::x_plot_interval])
x_labels = index.iloc[::x_plot_interval]
ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')

plt.plot()
plt.show()