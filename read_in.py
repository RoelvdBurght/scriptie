import pandas as pd

dfs = pd.ExcelFile("Input V1.xlsx")

df_bijzondere_dagen = pd.read_excel(dfs, "Bijzondere dagen")
df_marketing_campanges = pd.read_excel(dfs, "Marketing Campagne(s)")
df_weer = pd.read_excel(dfs, "Weather")
df_sea_clicks = pd.read_excel(dfs, "SEA Budget & Clicks")
df_budget_radio_tv = pd.read_excel(dfs, "TV_Radio Budget")
df_traffic = pd.read_excel(dfs, "Traffic")
df_emails = pd.read_excel(dfs, "Emails send & opens")
df_partner_vouchers = pd.read_excel(dfs, "Partner Voucher Orders")
df_omzet = pd.read_excel(dfs, "Omzet total en per Cat")
df_ordervolumes = pd.read_excel(dfs, "Ordervolumes total en per Cat")
df_sendvolumes = pd.read_excel(dfs, "Sendvolume total en per Cat")
df_send_is_order = pd.read_excel(dfs, "Senddate=Orderdate")

# Maak een lijst met de dataframes
df_list = [df_bijzondere_dagen, df_marketing_campanges, df_weer, df_sea_clicks,
			 df_budget_radio_tv, df_traffic, df_emails, df_partner_vouchers, df_omzet,
			 df_ordervolumes, df_sendvolumes, df_send_is_order]

# Standaardizeer de datums naar dd/mm/jjjj formaat
# ----------------------------
def convert(month):
	if month == "januari":
		return "01"
	if month == "februari":
		return "02"
	if month == "maart":
		return "03"
	if month == "april":
		return "04"
	if month == "mei":
		return "05"
	if month == "juni":
		return "06"
	if month == "juli" or month == "july":
		return "07"
	if month == "augustus":
		return "08"
	if month == "september":
		return "09"
	if month == "oktober":
		return "10"
	if month == "november":
		return "11"
	if month == "december":
		return "12"

def change_format(df):
	for index, date in enumerate(df.iloc[:, 0]):
		date_list = date.split()
		date_list[1] = convert(date_list[1])
		if len(date_list[0]) == 1:
			date_list[0] = "0" + date_list[0]
			# print(date_list)
		new_date = "-".join(date_list)
		df.iloc[index, 0] = new_date
	return df

def joe(df):
	for index, date in enumerate(df.iloc[:, 0]):
		date = str(date).split(" ")[0]
		date_list = date.split("-")
		if len(date_list[2]) == 1:
			date_list[2] = "0" + date_list[2]
		new_date_list = [date_list[2], date_list[1], date_list[0]]
		new_date = "-".join(new_date_list)
		df.iloc[index, 0] = new_date
	return df


for i, df in enumerate(df_list):
	print("df nummer ", i)
	try:
		cell = df.iloc[0, 0]
	except:
		cell = ""
	done = False
	try:
		for character in cell:
			if character.isalpha() and not done:
				df = change_format(df)
				done = True
		done = False
	except TypeError:
		df = joe(df)
	print(i)
	print(len(df.iloc[:, 0]))
# ------------------------------------

# Combineer dataframes met voorspellende variabelen tot combined_df




combined_df = pd.DataFrame({"Date" : df_sendvolumes.iloc[:, 0], "Weekday":df_sendvolumes.iloc[:, 1]})
skip = ["Date", "Weekday"]


for df in df_list[:-4]:
	count = 1
	collum_names = df.columns
	print(collum_names)
	print("Running for dataframe nr {}".format(count))
	for index, row in df.iterrows():
		for index_comb, row_comb in combined_df.iterrows():
			if row.iloc[0] == row_comb.iloc[0]:	
				print(row.iloc[0], row_comb.iloc[0])
				for col_name in collum_names:
					combined_df.loc[index_comb, col_name] = df.loc[index, col_name]
				# combined_df.loc[index_comb, ]
	
	count += 1
print(combined_df)
combined_df.to_csv("combined.csv", index=False)
print('printed')	
# for df in df_list[:-4]:
# 	print(df)
# 	columns = df.columns
# 	for index, row in df.iterrows():
# 		for col_name in columns:
# 			for index_combined, datum in enumerate(combined_df[col_name]):
# 				if datum == row[0]:
# 					combined_df.loc[index_combined, col_name] = df.loc[index, col_name]
# 		# zet df.loc[index, col_name] in combined_df waar df.iloc[index, 0] == combined_df.loc[index, "Date"]

# 		# 	if df.loc[index, col_name] in

# 		# if col_name not in skip:
# 		# 	for index, row in combined_df.iterrows():

# 	print(combined_df)
# 	# combined_df[col_name]
