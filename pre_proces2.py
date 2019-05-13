import pandas as pd
import string as st
import numpy as np

# def check_words(word_list):
#     return_list = []
#     for word in word_list:
#         if word is "dranken" or word == "bloemen" or word == "chocolade" or word == "kaarten" or word == "ballonnen" or word == "taarten" or word == "boeken":
#             return_list.append(word)
#         if word == "planten":
#             return_list.append("bloemen")
#         if word == "gebak" or word == "taart" or word == "gebakjes" or word == "taartjes":
#             return_list.append("taarten")
#         if word == "drank" or word == "bubbels":
#             return_list.append("dranken")
#     return set(return_list)
        

# comb = pd.read_csv("combined.csv")
# market = pd.read_excel("NEW_ Marketingkalender 2019.xlsx", sheet_name="2017")

# # Rowno is de rij waar de categories waar campagne op gevoerd word, handmatig aanpassen per sheet
# Rowno = 8

# # Maak een pandas Series met per dag volledige naam van de marketing campagne, dus exact wat er in de sheet staat
# campagnes = market.loc[8, :]
# maanddagen = market.loc[4, :]
# print(maanddagen.index[[0]])
# maanddagen = maanddagen.drop(index=maanddagen.index[[0]])
# maanddagen = maanddagen[:364]
# # maanddagen = maanddagen.drop(index=maandagen.index[[365:]]) #maanddagen.iloc[365:]
# print(maanddagen)
# print(market.loc[1,:])

# value = None
# temp_frame = pd.DataFrame()
# for index, cat in enumerate(campagnes):
#     if pd.notnull(cat) and cat is not value:
#         replace = True
#         value = cat
#     if replace:
#         campagnes[index] = value
# # print(campagnes)
# # s = "joe, isk advab&( dafs:; / kafh"
# # s = s.translate(str.maketrans('', '', string.punctuation))
# # print(s)
# # Vervang rauwe tekst door woorden
# for zin in campagnes:
#     # print(zin)
#     zin = zin.translate(str.maketrans('', '', st.punctuation)).lower()
#     word_list = zin.split(" ")
#     categories = []
#     if "multi" in word_list or "multi-categorie" in word_list or "multi-catgory" in word_list or "multi-cat" in word_list or "multicategory" in word_list:
#         # categories.append("multi-categorie")
#         pass
#     # categories = check_words(word_list)
#     joe = ",".join(categories)
#     # print(joe)The most robust and consistent way of slicing ranges along arbitrary axes is described in the Selection by Position section detailing the .iloc method. For now, we explain the semantics of slicing using the [] operator.



ar = np.asarray([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
print(ar)
print(ar.shape)

# ar2 = ar.reshape(2, -1, 1)
ar2 = ar.swapaxes(1,2)
print(ar2)
print(ar2.shape)

# data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# data = data.reshape((1, 1, 10))
# print(data)
# dfs = pd.ExcelFile("Input V1.xlsx")

# df_ordervolumes = pd.read_excel(dfs, "Ordervolumes total en per Cat")

# print(df_ordervolumes)