import pandas as pd
import numpy as np
from itertools import combinations
import math

class dataset(object):
    def __init__(self,trainPath,testPath,train=True):

        self.train_dataframe = pd.read_csv(trainPath,sep=',',index_col='id')
        self.test_dataframe = pd.read_csv(testPath,sep=',',index_col='id')
        self.train_label = self.train_dataframe['血糖']
        self.train = self.train_dataframe.iloc[:,0:-1]
        if train==True:
            self.test = self.test_dataframe.iloc[:,0:-1]
            self.test_label = self.test_dataframe['血糖']
        else:
            self.test = self.test_dataframe.iloc[:,:]
        self.filter_feature = False

    def trans_datetime2weather(self, weatherpath = "../tmp/weather.csv"):

        self.train["体检日期"] = pd.to_datetime(self.train["体检日期"])
        self.test["体检日期"] = pd.to_datetime(self.test["体检日期"])
        weatherdata = pd.read_csv(weatherpath, index_col='日期', parse_dates=True)

        self.train.insert(1, 'high_temperature', weatherdata.ix[self.train['体检日期'], :]['最高'].tolist())
        self.train.insert(1, 'low_temperature', weatherdata.ix[self.train['体检日期'], :]['最低'].tolist())

        self.test.insert(1, 'high_temperature', weatherdata.ix[self.test['体检日期'], :]['最高'].tolist())
        self.test.insert(1, 'low_temperature', weatherdata.ix[self.test['体检日期'], :]['最低'].tolist())

        self.train['diff_temperature'] = self.train['high_temperature'] - self.train['low_temperature']
        self.test['diff_temperature'] = self.test['high_temperature'] - self.test['low_temperature']

        self.train.drop(['体检日期'], axis=1, inplace=True)
        self.test.drop(['体检日期'], axis=1, inplace=True)

    def del_outlier(self):

        numerical_data = self.train[self.train.columns.drop(["性别",
                                                             '年龄',
                                                             'high_temperature',
                                                             'low_temperature',
                                                             'diff_temperature'])]
        # split for the outlier
        # 1.0 std function to Extract outliers
        std_outlier = numerical_data[(numerical_data - numerical_data.mean()) > 3 * numerical_data.std()]

        # 2.0 boxplot funciton to Extract outliers
        Q1 = numerical_data.quantile(0.25)
        Q3 = numerical_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        box_outlier = numerical_data[(numerical_data < Q1 - outlier_step) | (numerical_data > Q3 + outlier_step)]

        # 3.0 percent95 function to Extract outliers
        percent_outlier = numerical_data[
            (numerical_data > numerical_data.quantile(0.025)) | (numerical_data > numerical_data.quantile(0.975))]

        # vote
        outlier = numerical_data[((box_outlier.notnull()) & (std_outlier.notnull()))
                                 | ((box_outlier.notnull()) & (percent_outlier.notnull()))
                                 | ((std_outlier.notnull()) & (percent_outlier.notnull()))
                                 | ((box_outlier.notnull()) & (percent_outlier.notnull()) & (std_outlier.notnull()))
                                 ]
        # Extract outliers of outliers
        Q1_outlier = outlier.quantile(0.25)
        Q3_outlier = outlier.quantile(0.75)
        IQR_outlier = Q3_outlier - Q1_outlier
        outlier_outlier_step = 1.5 * IQR_outlier
        box_outlier_outlier = outlier[(outlier < Q1_outlier - outlier_outlier_step) |
                                      (outlier > Q3_outlier + outlier_outlier_step)]

        # the baseline to drop outliers
        outlier_baseline = box_outlier_outlier.min().fillna(float("inf"))
        # print(outlier_baseline)

        for i in numerical_data.columns:
            change = lambda x: outlier_baseline[i] if (x > outlier_baseline[i] )  else x
            self.train[i] = self.train[i].map(change)
            self.test[i] = self.test[i].map(change)

    def fill_nan(self):
        dataset = self.train.append(self.test)
        columns = (dataset.columns)
        columns = columns.drop(["性别",
                       "年龄",
                     'high_temperature',
                     'low_temperature',
                     'diff_temperature'])
        for i in columns:
            self.train[i] = self.train[i].fillna(dataset[i].mean())
            self.test[i] = self.test[i].fillna(dataset[i].mean())

    def generate_arithmetic(self):

        def polynomial(train, test, cols):
            dataset = train.append(test)
            for f1, f2 in list(combinations(cols, 2)):
                if self.filter_feature == True:
                    if f1 + '&' + f2 not in self.importance_f:
                        continue
                colx = train[f1] / dataset[f1].std()
                coly = train[f2] / dataset[f2].std()
                train[f1 + '&' + f2] = colx * coly

                colx = test[f1] / dataset[f1].std()
                coly = test[f2] / dataset[f2].std()
                test[f1 + '&' + f2] = colx * coly
            return train, test

        def difffeature(train, test, cols, name=''):
            dataset = self.train.append(self.test)
            for f1, f2 in list(combinations(cols, 2)):
                if self.filter_feature == True:
                    if f1 + '-' + f2 + name not in self.importance_f:
                        continue
                colx = (train[f1] - dataset[f1].mean()) / dataset[f1].std()
                coly = (train[f2] - dataset[f2].mean()) / dataset[f2].std()
                train[f1 + '-' + f2 + name] = colx - coly

                colx = (test[f1] - dataset[f1].mean()) / dataset[f1].std()
                coly = (test[f2] - dataset[f2].mean()) / dataset[f2].std()
                test[f1 + '-' + f2 + name] = colx - coly
            return train, test

        def addfeature(train, test, cols, w1=1, w2=1, name=''):
            dataset = self.train.append(self.test)
            for f1, f2 in list(combinations(cols, 2)):
                if self.filter_feature == True:
                    if f1 + '+' + f2 + name not in self.importance_f:
                        continue
                colx = (train[f1] - dataset[f1].mean()) / dataset[f1].std()
                coly = (train[f2] - dataset[f2].mean()) / dataset[f2].std()
                train[f1 + '+' + f2 + name] = w1 * colx + w2 * coly

                colx = (test[f1] - dataset[f1].mean()) / dataset[f1].std()
                coly = (test[f2] - dataset[f2].mean()) / dataset[f2].std()
                test[f1 + '+' + f2 + name] = w1 * colx + w2 * coly
            return train, test

        def divifeature(train, test, cols):
            dataset = self.train.append(self.test)
            for f1, f2 in list(combinations(cols, 2)):
                if self.filter_feature == True:
                    if f1 + '/' + f2 not in self.importance_f:
                        continue
                colx = train[f1] / dataset[f1].std()
                coly = train[f2] / dataset[f2].std()
                train[f1 + '/' + f2] = colx / coly
                # train[f2 + '/' + f1] = coly / colx

                colx = test[f1] / dataset[f1].std()
                coly = test[f2] / dataset[f2].std()
                test[f1 + '/' + f2] = colx / coly
                # test[f2 + '/' + f1] = coly / colx
            return train, test

        def quantile_feature(train, test, cols, groups=2):

            dataset = train.append(test)
            for col in cols:
                dataset['q_' + col] = pd.qcut(dataset[col], groups, labels=False)

            for f1, f2 in list(combinations(cols, 2)):
                dataset[f1 + '_q_' + f2] = groups * dataset['q_' + f1] + dataset['q_' + f2]
                train[f1 + '_q_' + f2] = dataset[f1 + '_q_' + f2].values[:train.shape[0]]
                test[f1 + '_q_' + f2] = dataset[f1 + '_q_' + f2].values[train.shape[0]:]
                train[f1 + '_q_' + f2] = train[f1 + '_q_' + f2].astype('category')
                test[f1 + '_q_' + f2] = test[f1 + '_q_' + f2].astype('category')
            return train, test
        contiCols = self.train.columns.drop(['high_temperature',
                                             'low_temperature',
                                             'diff_temperature',
                                             '性别',
                                             '白蛋白',
                                             '乙肝表面抗原',
                                             '乙肝表面抗体',
                                             '乙肝e抗原',
                                             '乙肝e抗体',
                                             '乙肝核心抗体',
                                             '嗜酸细胞%',
                                             '嗜碱细胞%'
                                             ])
        self.train, self.test = polynomial(self.train, self.test, contiCols)
        self.train, self.test = addfeature(self.train, self.test, contiCols)
        self.train, self.test = difffeature(self.train, self.test, contiCols)
        self.train, self.test = divifeature(self.train, self.test, contiCols)

    def category_sex(self):
        sex_dict = {"男": [1,0], "女": [0,1]}
        self.train = self.train.replace(sex_dict)
        self.test = self.test.replace(sex_dict)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def generate_matrix(self, train):
    #     if train == 1:
    #         X_matrix = self.data.iloc[:, :-1].values.astype(float)
    #         y_matrix = self.data.iloc[:, -1].values.astype(float)
    #     else:
    #         X_matrix = self.data.iloc[:, :].values.astype(float)
    #         y_matrix = np.array((len(self.data), 1))
    #     return X_matrix, y_matrix
    #
    #
    # def normalization0_1(self):
    #     """
    #     :return: normalization data to (0,1)
    #     """
    #     self.norm_data = (self.preprocess_data - self.preprocess_data.min()) / (self.preprocess_data.max() - self.preprocess_data.min())
    #     self.norm_feature = self.norm_data.iloc[:, :-1].values.astype(float)
    #     self.norm_label = self.norm_data.iloc[:, -1].values.astype(float)
    #     return self.norm_feature,self.norm_label
    #
    # def Z_score(self):
    #     """
    #     :return: normalization to mean:0,variance:1
    #     """
    #     self.norm_feature = (self.feature - self.feature.mean()) / (self.feature.std())
    #     self.norm_label = (self.label - self.label.mean()) / (self.feature.std())
    #     return self.norm_feature, self.norm_label
    #
    # def next_batch(self,batch_size):
    #     start = self.index_in_epoch
    #     self.index_in_epoch += batch_size
    #     if self.index_in_epoch > self.example_nums:
    #         # Finished epoch
    #         self.epochs_completed += 1
    #         # Shuffle the data
    #         perm = np.arange(self.example_nums)
    #         np.random.shuffle(perm)
    #         self.feature = self.feature[perm]
    #         self.label = self.label[perm]
    #         # Start next epoch
    #         start = 0
    #         self.index_in_epoch = batch_size
    #         assert batch_size <= self.example_nums
    #     end = self.index_in_epoch
    #     return np.array(self.feature[start:end]), np.array(self.label[start:end])
    #
    # def fill_nan(self):
    #     data = self.data
    #     sex_dict = {"男": 1, "女": 0}
    #     data = data.replace(sex_dict)
    #     data.fillna(round(data.mean(), 2), inplace=True)
    #     data = data.reset_index(drop=True)
    #     num_data = len(data)
    #     # data.mean().to_csv("../tmp/train_mean.csv")
    #     self.data = data
    #
    #
    #
    # def del_outlier(self):
    #
    #     data = self.data.reset_index(drop=True)
    #
    #     numerical_data = data[data.columns.drop(["性别", '年龄', "体检日期",'high_temperature','low_temperature','血糖'])]
    #     # split for the outlier
    #     #1.0 std function to Extract outliers
    #     std_outlier = numerical_data[(numerical_data - numerical_data.mean()) > 3 * numerical_data.std()]
    #
    #     #2.0 boxplot funciton to Extract outliers
    #     Q1 = numerical_data.quantile(0.25)
    #     Q3 = numerical_data.quantile(0.75)
    #     IQR = Q3 - Q1
    #     outlier_step = 1.5 * IQR
    #     box_outlier = numerical_data[(numerical_data < Q1 - outlier_step) | (numerical_data > Q3 + outlier_step)]
    #
    #     #3.0 percent95 function to Extract outliers
    #     percent_outlier = numerical_data[(numerical_data > numerical_data.quantile(0.025)) | (numerical_data > numerical_data.quantile(0.975))]
    #
    #     #vote
    #     outlier = numerical_data[((box_outlier.notnull()) & (std_outlier.notnull()))
    #                    | ((box_outlier.notnull()) & (percent_outlier.notnull()))
    #                    | ((std_outlier.notnull()) & (percent_outlier.notnull()))
    #                    | ((box_outlier.notnull()) & (percent_outlier.notnull()) & (std_outlier.notnull()))
    #                    ]
    #     #Extract outliers of outliers
    #     Q1_outlier = outlier.quantile(0.25)
    #     Q3_outlier = outlier.quantile(0.75)
    #     IQR_outlier = Q3_outlier - Q1_outlier
    #     outlier_outlier_step = 1.5 * IQR_outlier
    #     box_outlier_outlier = outlier[(outlier < Q1_outlier - outlier_outlier_step) |
    #                                   (outlier > Q3_outlier + outlier_outlier_step)]
    #
    #     #the baseline to drop outliers
    #     outlier_baseline = box_outlier_outlier.min()
    #
    #     for i in numerical_data.columns:
    #         change = lambda x: outlier_baseline[i] if x > outlier_baseline[i] and x != 'NaN' else x
    #         numerical_data[i] = numerical_data[i].map(change)
    #     data = numerical_data.join(data.ix[:,["性别", '年龄','high_temperature', 'low_temperature', '血糖']])
    #     self.data = data
    #     # return feature
    #
    # def add_feature(self,datetime=1,age_divide=0):
    #     if datetime == True:
    #         data = self.data
    #         data["体检日期"] = pd.to_datetime(data["体检日期"])
    #
    #         num_data = len(data)
    #         weatherpath = "../tmp/weather.csv"
    #         weatherdata = pd.read_csv(weatherpath, index_col='日期', parse_dates=True)
    #
    #         data.insert(1, 'high_temperature', weatherdata.ix[data['体检日期'], :]['最高'].tolist())
    #         data.insert(1, 'low_temperature', weatherdata.ix[data['体检日期'], :]['最低'].tolist())
    #
    #         self.data = data
    #
    #     if age_divide == 1:
    #         data = self.data
    #         bins = [0, 7, 18, 41, 66, 100]
    #         group_periods = [0, 1, 2, 3, 4]
    #         cats = pd.cut(data.年龄, bins, right=False, labels=group_periods)
    #         cats.rename('age_type')
    #         data.insert(1, 'age_type', cats.tolist())
    #         # print(data)
    #         self.data = data
    #         # data.to_csv("../tmp/addFeature1_onlinetest.csv", index_label='id')
    #
    # def generate_matrix(self,train):
    #     if train ==1:
    #         X_matrix = self.data.iloc[:, :-1].values.astype(float)
    #         y_matrix = self.data.iloc[:, -1].values.astype(float)
    #     else:
    #         X_matrix = self.data.iloc[:, :].values.astype(float)
    #         y_matrix = np.array((len(self.data),1))
    #     return X_matrix,y_matrix


# if age_divide == 1:
#     data = self.data
#     bins = [0, 7, 18, 41, 66, 100]
#     group_periods = [0, 1, 2, 3, 4]
#     cats = pd.cut(data.年龄, bins, right=False, labels=group_periods)
#     cats.rename('age_type')
#     data.insert(1, 'age_type', cats.tolist())
#     # print(data)
#     self.data = data
#     # data.to_csv("../tmp/addFeature1_onlinetest.csv", index_label='id')


        # if train==1:
        #     self.feature,self.label = self.data.iloc[:, :-1],self.data.iloc[:, -1]
        #     self.index_in_epoch = 0
        #     self.example_nums = len(self.label)
        #     self.epochs_completed = 0
        # else:
        #     self.feature = self.data.iloc[:, :]
        #     self.index_in_epoch = 0
        #     # self.example_nums = len(self.label)
        #     self.epochs_completed = 0