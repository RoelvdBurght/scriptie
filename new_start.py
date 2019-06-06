import pandas as pd 

def convert_day(day):
	''' Voegt een 0 aan de dagnummers 1 t/m 9 toe als dat nodig is
	'''
	if len(day) == 1:
		return('0' + day)
	else:
		return day

def convert_year(year):
	''' Converteerd yyyy naar yy format
	'''
	if len(year) == 4:
		return year[-2:]
	else:
		return year

def convert_month(month):
	''' Converteerd maand formatting van div formats naar mm numeriek    
	'''
	month = month.lower()
	if month == "januari" or month == 'jan' or month == '01' or month == '1':
		return "01"
	if month == "februari" or month == 'feb' or month == '02' or month == '2':
		return "02"
	if month == "maart" or month == 'mar' or month == '03' or month == '3':
		return "03"
	if month == "april" or month == 'apr' or month == '04' or month == '4':
		return "04"
	if month == "mei" or month == 'may' or month == '05' or month == '5':
		return "05"
	if month == "juni" or month == 'jun' or month == '06' or month == '6':
		return "06"
	if month == "juli" or month == "july " or month == 'jul' or month == '07' or month == '7':
		return "07"
	if month == "augustus" or month == 'aug' or month == '08' or month == '8':
		return "08"
	if month == "september" or month == 'sep' or month == '09' or month == '9':
		return "09"
	if month == "oktober" or month == 'oct' or month == '10' or month == '10':
		return "10"
	if month == "november" or month == 'nov' or month == '11' or month == '11':
		return "11"
	if month == "december" or month == 'dec' or month == '12' or month == '12':
		return "12"

def change_format(df, dic):
	''' De eerste kolom van dataframe df word omgezet naar dd-mm-yy formaat
		Als dic = 'old' wordt meegegeven word de lijst gereversed, in de oude
		data blijkt dat nodig
	'''
	date_series = df.iloc[:,0].astype(str)
	all_dates = []
	for date in date_series:
		# print(date)
		if '-' in date:
			date_list = date.split('-')
		if ' ' in date:
			date_list = date.split(' ')
		if '/' in date:
			date_list = date.split('/')
				
		date_list[1] = convert_month(date_list[1])
		if dic == 'old':
			date_list.reverse()
		date_list[0] = convert_day(date_list[0])
		date_list[2] = convert_year(date_list[2])
		all_dates.append('-'.join(date_list))
	df.iloc[:,0] = all_dates
	df = df.rename(columns={df.columns[0]: 'Date'})
	return df


# # Lees de data in uit de excel files
# input_old = pd.ExcelFile('Input V2 (more history).xlsx')
# input_new = pd.ExcelFile('Input V1.xlsx')

# # Maak dicts met dataframes aan, elke sheet is een value, naam is key
# df_dict_old = pd.read_excel(input_old, sheet_name=None)
# df_dict_new = pd.read_excel(input_new, sheet_name=None)

# # Verwijder onnodige sheets, ofwel leeg ofwel overbodig door nieuwe data (df_dict_old)
# del df_dict_new['Historie']
# del df_dict_new['Bijzondere dagen']
# del df_dict_new['Marketing Campagne(s)']
# del df_dict_new['SEA Budget & Clicks']
# del df_dict_new['Emails send & opens']
# del df_dict_new['TV_Radio Budget']

# # Verander de datums naar dd-mm-yyyy
# for df_name in df_dict_new:
# 	df = df_dict_new[df_name]
# 	df_fresh = change_format(df, 'new')
# 	df_dict_new[df_name] = df_fresh

# for df_name in df_dict_old:
# 	df = df_dict_old[df_name]
# 	df_fresh = change_format(df, 'old')
# 	df_dict_old[df_name] = df_fresh

# Sla de losse sheets op in nieuwe xlsx files
# writer1 = pd.ExcelWriter('input_1_clean_dates.xlsx')
# writer2 = pd.ExcelWriter('input_2_clean_dates.xlsx')

# for df_name, df in df_dict_new.items():
# 	df.to_excel(writer1, df_name)

# for df_name, df in df_dict_old.items():
# 	df.to_excel(writer2, df_name)
# writer1.save()
# writer2.save()


# ------------- handmatig de kolommen onder elkaar gezet waar nodig
# Nu de kolommen in 1 csv zetten

# # load data
# excel = pd.ExcelFile('new_clean_sheets.xlsx')
# df_dict = pd.read_excel(excel, sheet_name=None)
# datums = df_dict['Weather']['Date']

# # Set dates as index
# for name, dic in df_dict.items():
# 	dic.set_index('Date', drop=True, inplace=True)

# # Base df is the final dataframe where all comes together
# base_df = pd.DataFrame()
# base_df['Date'] = datums
# base_df.set_index('Date', drop=True, inplace=True)
# base_df['Target'] = df_dict['Sendvolume total en per Cat']['Grand Total']
# base_df['Weekday'] = df_dict['Weather']['Weekday']
# base_df['Avg temp'] = df_dict['Weather']['Avg. Temperature']
# base_df['Sunshine_minutes'] = df_dict['Weather']['sunshine_duration_in_min']
# base_df['Rain_minutes'] = df_dict['Weather']['rain_time_in_min']
# base_df['Voucher orders'] = df_dict['Voucher Orders total']['Grand Total']
# base_df['Traffic'] = df_dict['Traffic']['Users']
# base_df['SEA Cost'] = df_dict['SEA Budget & Clicks']['Som van Kosten']
# base_df['SEA Clicks'] = df_dict['SEA Budget & Clicks']['Som van Klikken']
# base_df['Emails accepted'] = df_dict['Email send en opens']['aantal_geaccepterrd']
# base_df['Emails open'] = df_dict['Email send en opens']['unieke_opens']
# base_df.Weekday = base_df.Weekday.str.lower()
# base_df.to_csv('raw_data_v3.csv')
# Handmatig speciale dagen toegevoegd in excel

# ----------------------- One hot encoden en fillna en enkel de data vanaf 

# df = pd.read_csv('raw_data_v3.csv')
# df.set_index('Date', drop=True, inplace=True)
# df = pd.get_dummies(df, columns=['Weekday', 'Spc dag pos aanloop', 'Spc dag pos', 'Spc dag neg'])
# df.fillna(0, inplace=True)
# df = df.loc[:'01-04-19',:]
# df.to_csv('clean_v3_2012.csv')
# df = df.loc['01-01-13':,:]
# df.to_csv('clean_v3_2013.csv')
