import pandas as pd
import numpy as np 


tmp = pd.read_csv('tmp.csv')
def convert(month):
	month = month.lower()
	print(month)
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
	if month == "oktober":
		return "10"
	if month == "november":
		return "11"
	if month == "december":
		return "12"

def change_format(df, col_num):
	for index, date in enumerate(df.iloc[:, col_num]):
		print(date)
		date_list = date.split("/")
		print(date_list)
		date_list[1] = convert(date_list[1])
		if len(date_list[0]) == 1:
			date_list[0] = "0" + date_list[0]
			# print(date_list)
		print(date_list)
		new_date = "-".join(date_list)
		df.iloc[index, 0] = new_date
	return df

# print(tmp.iloc[:, 0].isnull())
jool_arr = tmp.iloc[:, 0].isnull()
bool_arr = []
for i in jool_arr:
	bool_arr.append(not(i))

# print(bool_arr)

dates_long = tmp.iloc[bool_arr, 0]

# print(dates_long)
dates_short = tmp.iloc[:, 1]
# print(dates_short)
# print(tmp.loc[:, 'untitled'] == np.nan)


# -------------------------------------------------
final = pd.read_csv('test_final.csv')

tmp = pd.read_csv('tmp.csv')
cols_f = final.columns
cols_t = tmp.columns
print(cols_t)
date = final.loc[:, cols_f[0]]

date2 = tmp.loc[:, cols_t[2]]
date = pd.to_datetime(date)
# date = date.dt.date

date2 = pd.to_datetime(date2)
# date2 = date2.dt.date
# final.loc[]
# print(date)
# print(date2)
count = 0
for index, datum2 in date2.iteritems():
	# Waar de datum in series date hetzelfde is als datum_email
	# Zet op die index in dataframe final de bijbehorende values die in tmp staan
	index_new = date.index[date.loc[:] == datum2]
	final.loc[index_new, 'Ordered=Send'] = tmp.loc[index, cols_t[4]]
	
	# final.loc[index_emails, 'Emails unieke opens'] = tmp.loc[index, cols_t[2]]

print(final)
final.to_csv("testjeeeuh.csv")
# --------------------------------------------------------

# tmp = pd.read_csv('tmp.csv')
# tmp.loc[:, 'fake_data'] = pd.to_datetime(tmp.loc[:,'fake_data'])
# tmp.loc[:, 'real_data'] = pd.to_datetime(tmp.loc[:, 'real_data'])

# print(tmp.columns)
# print(tmp[tmp.loc[:, 'fake_data'] != tmp.loc[:, 'real_data']])