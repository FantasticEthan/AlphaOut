import pandas as pd
import numpy as np
from itertools import combinations
import math

class dataset(object):
    def __init__(self,trainPath,testPath,trainable=True):

        self.train_dataframe = pd.read_csv(trainPath,sep=',',index_col='id')
        self.test_dataframe = pd.read_csv(testPath,sep=',',index_col='id')
        self.train_label = self.train_dataframe['血糖']
        self.train = self.train_dataframe.iloc[:,0:-1]

        self.flag = trainable
        if self.flag==True:
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

    def fillna_outliermean(self):

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
        # Q1_outlier = outlier.quantile(0.25)
        # Q3_outlier = outlier.quantile(0.75)
        # IQR_outlier = Q3_outlier - Q1_outlier
        # outlier_outlier_step = 1.5 * IQR_outlier
        # box_outlier_outlier = outlier[(outlier < Q1_outlier - outlier_outlier_step) |
        #                               (outlier > Q3_outlier + outlier_outlier_step)]

        outlier_baseline = outlier.min()
        mean_fill =numerical_data[numerical_data<outlier_baseline].mean()

        # the baseline to drop outliers
        # outlier_baseline = box_outlier_outlier.min().fillna(float("inf"))
        # print(outlier_baseline)

        for i in numerical_data.columns:
            # change = lambda x: outlier_baseline[i] if (x > outlier_baseline[i] )  else x
            self.train[i] = self.train[i].fillna(mean_fill[i])
            self.test[i] = self.test[i].fillna(mean_fill[i])

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
            dataset = train.append(test)
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
            dataset = train.append(test)
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
            dataset = train.append(test)
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

    def liver_columns(self):
        columns = ['high_temperature',
                   'low_temperature',
                   'diff_temperature',
                   '性别',
                   '*天门冬氨酸氨基转换酶',
                   '*丙氨酸氨基转换酶',
                   '*碱性磷酸酶',
                   '*r-谷氨酰基转换酶',
                   '*总蛋白',
                   '白蛋白',
                   '*球蛋白',
                   '白球比例']

        self.liver_train = self.train[columns].dropna()
        self.liver_train_label = self.train_label.loc[self.liver_train.index]

        if self.flag == 1:
            self.liver_test = self.test[columns].dropna()
            self.liver_test_label = self.test_label.loc[self.liver_test.index]
        else:
            self.liver_test = self.test[columns].dropna()
            self.liver_test_label = pd.DataFrame(index=self.liver_test.index)
        return self.liver_train,self.liver_train_label,self.liver_test,self.liver_test_label

    def bloodfat_columns(self):
        columns = ['high_temperature',
                   'low_temperature',
                   'diff_temperature',
                   '性别',
                   '甘油三酯',
                   '总胆固醇',
                   '高密度脂蛋白胆固醇',
                   '低密度脂蛋白胆固醇']

        self.bloodfat_train = self.train[columns].dropna()
        self.bloodfat_train_label = self.train_label.loc[self.bloodfat_train.index]

        if self.flag == 1:
            self.bloodfat_test = self.test[columns].dropna()
            self.bloodfat_test_label = self.test_label.loc[self.bloodfat_test.index]
        else:
            self.bloodfat_test = self.test[columns].dropna()
            self.bloodfat_test_label = pd.DataFrame(index=self.bloodfat_test.index)

        return self.bloodfat_train, self.bloodfat_train_label, self.bloodfat_test, self.bloodfat_test_label

    def urea_columns(self):
        columns = ['high_temperature',
                   'low_temperature',
                   'diff_temperature',
                   '性别',
                   '尿素',
                   '肌酐',
                   '尿酸']

        self.urea_train = self.train[columns].dropna()
        self.urea_train_label = self.train_label.loc[self.urea_train.index]

        if self.flag == 1:
            self.urea_test = self.test[columns].dropna()
            self.urea_test_label = self.test_label.loc[self.urea_test.index]
        else:
            self.urea_test = self.test[columns].dropna()
            self.urea_test_label = pd.DataFrame(index=self.urea_test.index)

        return self.urea_train, self.urea_train_label, self.urea_test, self.urea_test_label

    def hepatitis(self):
        columns = ['high_temperature',
                   'low_temperature',
                   'diff_temperature',
                   '性别',
                   '乙肝表面抗原',
                   '乙肝表面抗体',
                   '乙肝e抗原',
                   '乙肝e抗体',
                   '乙肝核心抗体',]

        self.hepatitis_train = self.train[columns].dropna()
        self.hepatitis_train_label = self.train_label.loc[self.hepatitis_train.index]

        if self.flag == 1:
            self.hepatitis_test = self.test[columns].dropna()
            self.hepatitis_test_label = self.test_label.loc[self.hepatitis_test.index]
        else:
            self.hepatitis_test = self.test[columns].dropna()
            self.hepatitis_test_label = pd.DataFrame(index=self.hepatitis_test.index)

        return self.hepatitis_train, self.hepatitis_train_label, self.hepatitis_test, self.hepatitis_test_label

    def bloodnorm(self):
        columns = ['high_temperature',
                   'low_temperature',
                   'diff_temperature',
                   '性别',
                   '白细胞计数',
                   '红细胞计数',
                   '血红蛋白',
                   '红细胞压积',
                   '红细胞平均体积',
                   '红细胞平均血红蛋白量',
                   '红细胞平均血红蛋白浓度',
                   '红细胞体积分布宽度',
                   '血小板计数',
                   '血小板平均体积',
                   '血小板体积分布宽度',
                   '血小板比积',
                   '中性粒细胞%',
                   '淋巴细胞%',
                   '单核细胞%',
                   '嗜酸细胞%',
                   '嗜碱细胞%']

        self.bloodnorm_train = self.train[columns].dropna()
        self.bloodnorm_train_label = self.train_label.loc[self.bloodnorm_train.index]

        if self.flag == 1:
            self.bloodnorm_test = self.test[columns].dropna()
            self.bloodnorm_test_label = self.test_label.loc[self.bloodnorm_test.index]
        else:
            self.bloodnorm_test = self.test[columns].dropna()
            self.bloodnorm_test_label = pd.DataFrame(index=self.bloodnorm_test.index)

        return self.bloodnorm_train, self.bloodnorm_train_label, self.bloodnorm_test, self.bloodnorm_test_label

    def add_data(self,addPath="../tmp/createData2.csv"):
        self.df_add = pd.read_csv(addPath,encoding='gb2312',index_col='id')
        self.train_dataframe = self.train.join(self.train_label).append(self.df_add).reset_index(drop=True)
        self.train_label = self.train_dataframe['血糖']
        self.train = self.train_dataframe.iloc[:, 0:-1]

