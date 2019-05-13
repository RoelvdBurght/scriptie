from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pandas as pd 

# Lees data in en pak target kolom appart
data = pd.read_csv('combined.csv')
# target = data.loc[:, 'Grand Total']
# data.drop('Grand Total', axis=1, inplace=True)

# Maak van de datum de indexes
data.set_index('Date', inplace=True)
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pandas as pd 
# Drop speciale dag want dat is nog niet compleet
data.drop('Speciale_dag', axis=1, inplace=True)

# Pak de data tussen 01-01-2018 en 01-04-2019, dit is alles waar niks van mist
data_test = data.loc['01-01-2018':'01-04-2019', :]

# One hot encode de kolom multi-cat_type en drop hem
for index, row in data_test.iterrows():
    cats = row['Multi-cat_type']
    if not pd.isnull(cats):
        cats = cats.lower()
        cats = cats.replace(" ", '')
        cats = cats.split(',')
        for cat in cats:
            colname = 'multi_cat_type_' + cat
            data_test.loc[index, colname] = 1
data_test.drop(['Multi-cat_type'], inplace=True, axis=1)

# One hot encode marketing campagnes en vervang overal Nan door 0
data_test = pd.get_dummies(data_test, columns=['Marketing_campagne_cat'], prefix='marketing')
data_test.fillna(0, inplace=True)

# One hot de weekdagen
data_test = pd.get_dummies(data_test, columns=['Weekday'])

# Sla schone data op
data_test.to_csv('clean_v1.csv')
