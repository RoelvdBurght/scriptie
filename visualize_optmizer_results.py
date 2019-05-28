import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pprint as pp

def niceify_boxplot(bp):

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    return bp



with open('out3.txt') as json_file:
    data = json.load(json_file)

all_data = {}
model_names = []
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
    model_names.append(key)
    i += 1

mse_boxes = []
mae_boxes = []
loss_train = []
loss_val = []
for model in model_names:
    mse_boxes.append(all_data[model]['mse_test'])
    mae_boxes.append(all_data[model]['mae_test'])
    loss_train.append(all_data[model]['loss'])
    loss_val.append(all_data[model]['val_loss'])
    # for iteration in all_data[model]['loss']:
    #     loss_train.append(iteration)
    # for iteration in all_data[model]['val_loss']:
    #     loss_val.append(iteration)


avg_mse = np.average(mse_boxes)
avg_mae = np.average(mae_boxes)

# fig1 = plt.figure(1, figsize=(10,10))
# ax1 = fig1.add_subplot(211)
# ax1.axhline(avg_mse, c='g')
# bp = ax1.boxplot(mse_boxes,  patch_artist=True)
# bp = niceify_boxplot(bp)

# ax2 = fig1.add_subplot(212)
# bp = ax2.boxplot(mae_boxes, patch_artist=True)
# bp = niceify_boxplot(bp)
# ax2.axhline(avg_mae, c='g')
# x_ax = []
# for i in range(len(mse_boxes)):
#     x_ax_inner = []
#     for jippie in mse_boxes[i]:
#         x_ax_inner.append(i)
#     x_ax.append(x_ax_inner)

# fig_scat = plt.figure(3, figsize=(10,10))
# ax1 = fig_scat.add_subplot(211)
# for ind, model_mse in enumerate(mse_boxes):
#     ax1.scatter(x_ax[ind], model_mse)

# ax1.scatter(x_ax,mse_boxes)
# ax2 = fig_scat.add_subplot(221)
# ax2.scatter(x_ax,mae_boxes)

# fig_loss = plt.figure(2, figsize=(10,10))
# pos = 0
# for model in model_names:
#     pos += 1
#     coordinate = int('24' + str(pos))
#     ax3 = fig_loss.add_subplot(coordinate)
#     for loss in all_data[model]['loss']:
#         epochs = range(len(loss))
#         ax3.plot(epochs, loss)
#         # plt.ylim(0, 10000000)
#     for val_loss in all_data[model]['val_loss']:
#         ax3.plot(epochs, val_loss)

# for model in model_names:
#     for iteration, loss in enumerate(all_data[model]['loss']):
#         pos +=1
#         # coordinate = (5,4,pos)
#         ax3 = fig_loss.add_subplot(4,5, pos)
#         epochs = range(len(loss))
#         plt.plot(epochs, loss)
#         plt.plot(epochs, all_data[model]['val_loss'][iteration])
#         plt.ylabel(model)

# fig_loss_val = plt.figure(3, figsize=(10,10))
# pos = 0
# for model in model_names:
#     pos += 1
#     coordinate = int('24' + str(pos))
#     ax3 = fig_loss_val.add_subplot(coordinate)
#     for loss in all_data[model]['val_loss']:
#         epochs = range(len(loss))
#         ax3.plot(epochs, loss)
#         # plt.ylim(10000000, 1000000000)
plt.show()