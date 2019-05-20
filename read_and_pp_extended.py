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
cut_outliers = True

# # Verwijder target kolom
# target = data.loc[:, 'Grand Total']
# data.drop('Grand Total', axis=1, inplace=True)

# Maak van de datum de indexes
data.set_index('Date', inplace=True)
print(data.index)
# Pak de data tussen 01-01-2017 en 01-04-2019
data_test = data.loc['01-01-12':'01-04-19', :]
data_test.drop(labels=['Emails geaccepteerd', 'Emails unieke opens', 'Traffic split desktop',
                 'Traffic split mobile', 'Traffic split tablet', 'Marketing_campagne_cat', 'Multi-cat_type', 'TV', 'ATL - Media', 'Radio'], axis=1, inplace=True)
data_test = pd.get_dummies(data_test, columns=['Spc dag pos aanloop',	'Spc dag pos', 'Spc dag neg', 'Weekday'])

# clean de foktop dingen in temp en regen
data_test.loc[:, 'Avg. Temperature'] = data_test.loc[:, 'Avg. Temperature'].str.strip(' ,')
new_col = []
for joe in data_test.loc[:, 'Avg. Temperature']:
	if isinstance(joe, str) and len(joe) == 3:
		joe = '-' + joe[1]
		if joe[1] == '0':
			joe = 0
	new_col.append(int(joe))

data_test['Avg. Temperature'] = new_col
# data_test['rain_time_in_min'] = pd.to_numeric(data_test['rain_time_in_min'])
c = 0
new_col = []
for joe in data_test['rain_time_in_min']:
	if len(joe) > 3:
		print('--------------------------------------', c)
		print(joe)
		# data_test.loc[joe, 'rain_time_in_min'] = joe.replace(',', '')
		joe = int(joe.replace(',', ''))
	c+=1
	new_col.append(int(joe))

data_test['rain_time_in_min'] = new_col
# for joe in data_test['rain_time_in_min']:
# 	if len(joe) > 3:
# 		print('--------------------------------------', c)
# 		print(joe)
# 	c+=1

# data_test['rain_time_in_min'] = data_test['rain_time_in_min'].astype(int)


# Haalt december en de week voor valentijnsdag uit de data
if cut_outliers:
	valendays = ['08', '09', '10', '11', '12', '13', '14']
	# data_test = data_test[data_test.loc[:,'Spc dag pos aanloop_Kerst'] == 0]
	# data_test = data_test[data_test.loc[:, 'Spc dag pos aanloop_Valentijnsdag'] == 0]
	# data_test.drop(labels=['Spc dag pos aanloop_Kerst', 'Spc dag pos aanloop_Valentijnsdag'], axis=1)
	indices = data_test.index
	bool_bois = []
	for i in indices:
		joe = True
		split = i.split('-')
		day = split[0]
		month = split[1]
		if month == '12' or (month == '02' and day in valendays):
			joe = False
		bool_bois.append(joe)
	data_test = data_test[bool_bois]
	data_test.drop('12-07-18') # drop die gekke dag waarop niks verkocht is


# Sla schone data op
data_test.to_csv('clean_v2_extended_no_outliers.csv')
