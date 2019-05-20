import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

class Plotter:
    def __init__(self, data_size, data_set, look_back):
        self.data_size = data_size
        self.data_set = data_set
        self.look_back = look_back
        # Lees data in en sla de index en column labels op
        self.data = self.read_data()
        self.index = self.data.loc[:, 'Date']
        columns = list(self.data.columns)
        columns.remove('Date')
        self.columns = columns


        # Drop de rijen waar nan in voorkomt en de kolom met datums
        self.data.dropna(how='any', inplace=True)
        self.data = self.data.drop(labels='Date', axis=1)

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
        return data

    def plot_all_sales(self):
        if self.data_size == 's':
            x_plot_interval = 20
            fig = plt.figure(1, figsize=(8,8))
            ax1 = fig.add_subplot(111)
            x_axis = np.arange(0, len(self.Y_data), 1)

            ax1.plot(x_axis, self.Y_data)
            ax1.xaxis.set_ticks(x_axis[::x_plot_interval])
            x_labels = self.index.iloc[::x_plot_interval]
            ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')

            plt.plot()
            plt.show()

        if self.data_size == 'l':
            x_plot_interval = 50
            fig = plt.figure(1, figsize=(8,8))
            ax1 = fig.add_subplot(111)
            x_axis = np.arange(0, len(self.Y_data), 1)

            ax1.plot(x_axis, self.Y_data)
            ax1.xaxis.set_ticks(x_axis[::x_plot_interval])
            x_labels = self.index.iloc[::x_plot_interval]
            ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
            ax1.tick_params(axis='x', labelsize='7', labelright=True)


            plt.plot()
            plt.show()
    
    # def plot_sales_vs(feature):
    #     x_axis  = self.data[feature]
    #     if self.data_size == 's':
    #         x_plot_interval = 20
    #         fig = plt.figure(1, figsize=(8,8))
    #         ax1 = fig.add_subplot(111)

    #         ax1.plot(x_axis, self.Y_data)
    #         ax1.xaxis.set_ticks(x_axis[::x_plot_interval])
    #         x_labels = self.index.iloc[::x_plot_interval]
    #         ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')

    #         plt.plot()
    #         plt.show()

    #     if self.data_size == 'l':
    #         x_plot_interval = 50
    #         fig = plt.figure(1, figsize=(8,8))
    #         ax1 = fig.add_subplot(111)
    #         x_axis = np.arange(0, len(self.Y_data), 1)

    #         ax1.plot(x_axis, self.Y_data)
    #         ax1.xaxis.set_ticks(x_axis[::x_plot_interval])
    #         x_labels = self.index.iloc[::x_plot_interval]
    #         ax1.xaxis.set_ticklabels(x_labels, rotation=30, ha='right')
    #         ax1.tick_params(axis='x', labelsize='7', labelright=True)


    #         plt.plot()
    #         plt.show()
    
    def plot_poly_regression(self, split):
        self.split_data(split)
        reg = LinearRegression().fit(self.X_train, self.Y_train)
        print(reg.score(self.X_train, self.Y_train))
        print(self.Y_test.shape)
        print(self.Y_train.shape)

        results = reg.predict(self.X_test)
        x_axis = range(len(results))
        # df = pd.DataFrame(data=reg.coef_, columns=self.columns)
        # print(df)
        plt.plot(x_axis, results, c='r')
        plt.plot(x_axis, self.Y_test, c='g')
        plt.show()


data_size = 'l'
data_set = 'reg'
look_back = 7
set_split = 0.7

plotter = Plotter(data_size, data_set, look_back)
# plotter.plot_all_sales()
plotter.plot_poly_regression(set_split)