import pandas as pd
import numpy as np 

def convert(month):
	month = month.lower()
	print(month)
def convert(month):
	month = month.lower()
	if month == "januari" or month == 'jan':
		return "01"
	if month == "februari" or month == 'feb':
		return "02"
	if month == "maart" or month == 'mar':
		return "03"
	if month == "april" or month == 'apr':
		return "04"
	if month == "mei" or month == 'may':
		return "05"
	if month == "juni" or month == 'jun':
		return "06"
	if month == "juli" or month == "july " or month == 'jul':
		return "07"
	if month == "augustus" or month == 'aug':
		return "08"
	if month == "september" or month == 'sep':
		return "09"
	if month == "oktober" or month == 'oct':
		return "10"
	if month == "november" or month == 'nov':
		return "11"
	if month == "december" or month == 'dec':
		return "12"

def change_format(df):
	for index, date in enumerate(df.iloc[:, 0]):
		# print(date)
		if pd.isnull(date):
			return df
		date_list = date.split('-')
		date_list[1] = convert(date_list[1])
		if len(date_list[0]) == 1:
			date_list[0] = "0" + date_list[0]
		new_date = "-".join(date_list)
		df.iloc[index, 0] = new_date
	return df


# Lees data in en pak target kolom appart
data = pd.read_csv('raw_data_v2.csv')
dates = change_format(data)

# # Verwijder target kolom
# target = data.loc[:, 'Grand Total']
# data.drop('Grand Total', axis=1, inplace=True)

# Maak van de datum de indexes
data.set_index('Date', inplace=True)
print(data.index)
# Pak de data tussen 01-01-2017 en 01-04-2019
data_test = data.loc['01-01-17':'01-04-19', :]

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

# One hot de bijzondere dagen en aanloop
data_test = pd.get_dummies(data_test, columns=['Spc dag pos aanloop', 'Spc dag pos', 'Spc dag neg'])

# Drop de marketing budgetten in afwachting van nieuwe betere en verder teruggaande data
data_test = data_test.drop(['TV', 'Radio', 'ATL - Media'], axis=1)
data_test.loc[:, 'Avg. Temperature'] = data_test.loc[:, 'Avg. Temperature'].str.strip('() ,')
data_test['rain_time_in_min'] = pd.to_numeric(data_test['rain_time_in_min'])
c = 0
for joe in data_test['rain_time_in_min']:
	if len(joe) > 3:
		print('--------------------------------------', c)
		print(joe)
		data_test.loc[joe, 'rain_time_in_min'] = joe.replace(',', '')
	c+=1
for joe in data_test['rain_time_in_min']:
	if len(joe) > 3:
		print('--------------------------------------', c)
		print(joe)
	c+=1

data_test['rain_time_in_min'] = data_test['rain_time_in_min'].astype(int)


# Sla schone data op
data_test.to_csv('clean_v2.csv')
