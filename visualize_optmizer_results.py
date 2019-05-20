import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


with open('out.txt') as json_file:
    data = json.load(json_file)

all_data = {}
i = 1
for model in data:
    key = 'm' + str(i)
    model_boi = {}
    mse_list, mae_list, mae_train_list, mse_train_list, loss_list, val_loss_list = [],[],[],[],[],[]
    for iteration in model:
        mse, mae, mse_train, mae_train, loss, val_loss = iteration
        mse_list.append(mse)
        mae_list.append(mae)
        mse_train_list.append(mse_train)
        mae_train_list.append(mae_train)
        loss_list.append(loss)
        val_loss_list.append(val_loss)

    model_boi['mse_test'] = mse_list
    model_boi['mae_test'] = mae_list
    model_boi['mse_train'] = mse_train_list
    model_boi['mae_train'] = mae_train_list
    model_boi['loss'] = loss_list
    model_boi['val_loss'] = val_loss_list

    all_data[key] = model_boi
    i += 1

print(all_data)
# df = pd.DataFrame(data=[mse_list, mae_list, mse_train_list, mae_train_list], columns=['mse_test', 'mae_test', 'mse_train', 'mae_train'])
# print(df)