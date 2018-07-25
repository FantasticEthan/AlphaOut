import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import choice
import data_helper
import time

trainPath = "../tmp/train_all.csv"
localTestPath = "../tmp/localtest.csv"
onlineTestPath = "../data/d_train_20180102.csv"

dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/localtest.csv",trainable=1)

dataset.trans_datetime2weather()
dataset.category_sex()
dataset.fillna_outliermean()
X_train,y_train = dataset.train,dataset.train_label
X_train['血糖'] = y_train

less_data = X_train[(X_train['血糖']>6.1) |(X_train['血糖']<22)]

less_data = less_data.reset_index(drop=True)
index_new = less_data.columns
# mean_fill =numerical_data[numerical_data<outlier_baseline].mean()
# less_data = np.array(
#     data[data.iloc[:, tag_index] == np.array(case_state[case_state == min(case_state)].index)[0]])
# more_data = np.array(
#     data[data.iloc[:, tag_index] == np.array(case_state[case_state == max(case_state)].index)[0]])
# 找出每个少量数据中每条数据k个邻居
neighbors = NearestNeighbors(n_neighbors=5).fit(less_data)
location = []
for i in range(len(less_data)):
    location_set = neighbors.kneighbors([less_data.ix[i,:]], return_distance=False)[0]
    print(location_set)
    exit()
    location.append(location_set)

# 初始化，判断连续还是分类变量采取不同的生成逻辑
times = 0
continue_index = []  # 连续变量
class_index = []  # 分类变量
for i in (less_data.columns):
    if len(pd.DataFrame(less_data.ix[:, i]).drop_duplicates()) > 18:
        continue_index.append(i)
    else:
        class_index.append(i)
case_update = list()
location_transform = np.array(location)
method = 'random'
print(continue_index)
print(class_index)

while times < 3000:
    # 连续变量取附近k个点的重心，认为少数样本的附近也是少数样本
    pool = np.random.permutation(len(location))[1]
    neighbor_group = location_transform[pool]
    if method == 'mean':
        new_case1 = less_data.ix[list(neighbor_group), continue_index].mean()
    # 连续样本的附近点向量上的点也是异常点
    if method == 'random':
        away_index = np.random.permutation(len(neighbor_group) - 1)[1]
        neighbor_group_removeorigin = neighbor_group[1:][away_index]
        new_case1 = less_data.ix[pool,continue_index] + np.random.rand() * abs(
        less_data.ix[pool,continue_index] - less_data.ix[neighbor_group_removeorigin,continue_index])
    # 分类变量取mode
    #     print(new_case1)
    # mode = less_data.ix[neighbor_group, class_index].mode()
    # exit()
    new_case2 = less_data.ix[neighbor_group, class_index].mode().iloc[0, :]
    new_case2 = new_case2.fillna(less_data.ix[neighbor_group[0], class_index])
    new_case = pd.concat([new_case1,new_case2])

    if times == 0:
        case_update = pd.DataFrame(columns=index_new)
        case_update.loc[0] = new_case
    else:
        case_update.loc[times] = new_case
    # print(case_update)
    print('已经生成了%s条新数据，完成百分之%.2f' % (times, times * 100 / 3000))
    times = times + 1
    # time.sleep(0.1)
    # exit()
print(case_update.describe())

case_update.to_csv("../tmp/createData2.csv")
exit()

less_origin_data = np.hstack((less_data[:, continue_index], less_data[:, class_index]))
more_origin_data = np.hstack((more_data[:, continue_index], more_data[:, class_index]))
data_res = np.vstack((more_origin_data, less_origin_data, np.array(case_update.T)))
label_columns = [0] * more_origin_data.shape[0] + [1] * (
less_origin_data.shape[0] + np.array(case_update.T).shape[0])
data_res = pd.DataFrame(data_res)
# return data_res