import pandas as pd
from itertools import combinations
# import add_Data


class dataset(object):
    def __init__(self,trainPath,testPath,test=False):
        usecol=[
        'id',
        '性别',
        '年龄',
        '体检日期',
        '*天门冬氨酸氨基转换酶',
        '*丙氨酸氨基转换酶',
        '*碱性磷酸酶',
        '*r-谷氨酰基转换酶',
        '*总蛋白',
        '白蛋白',
        '*球蛋白',
        '白球比例',
        '甘油三酯',
        '总胆固醇',
        '高密度脂蛋白胆固醇',
        '低密度脂蛋白胆固醇',
        '尿素',
        '肌酐',
        '尿酸',

        '乙肝表面抗原',
        '乙肝表面抗体',
        '乙肝e抗原',
        '乙肝e抗体',
        '乙肝核心抗体',

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
        '血糖',
        ]
        self.train_dataframe = pd.read_csv(trainPath,usecols=usecol,index_col='id')
        self.train_label = self.train_dataframe['血糖']
        self.train = self.train_dataframe.iloc[:,0:-1]
        
        if test==False:
            self.test_dataframe = pd.read_csv(testPath,usecols=usecol,index_col='id')
            self.test = self.test_dataframe.iloc[:,0:-1]
            self.test_label = self.test_dataframe['血糖']
        else:
            self.test_dataframe = pd.read_csv(testPath,usecols=usecol[0:-1],index_col='id')
            self.test = self.test_dataframe.iloc[:,:]
        self.filter_feature = True
        self.importance_f = ['diff_temperature',
                             'high_temperature',
                             '年龄+*丙氨酸氨基转换酶',
                             'low_temperature',
                             '红细胞平均体积+红细胞体积分布宽度',
                             '年龄-红细胞体积分布宽度',
                             '尿酸-血红蛋白',
                             '*天门冬氨酸氨基转换酶/甘油三酯',
                             '年龄+红细胞平均血红蛋白浓度',
                             '年龄+红细胞计数',
                             '尿酸-红细胞计数',
                             '年龄&白细胞计数',
                             '年龄+尿素',
                             '年龄+白细胞计数',
                             '年龄&*r-谷氨酰基转换酶',
                             '*天门冬氨酸氨基转换酶/总胆固醇',
                             '年龄&甘油三酯',
                             '红细胞平均体积&红细胞体积分布宽度']
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

        outlier_baseline = outlier.min()
        mean_fill = numerical_data[numerical_data < outlier_baseline].mean()
        for i in numerical_data.columns:
            # change = lambda x: outlier_baseline[i] if (x > outlier_baseline[i] )  else x
            self.train[i] = self.train[i].fillna(mean_fill[i])
            self.test[i] = self.test[i].fillna(mean_fill[i])

    def generate_arithmetic(self):
        #将每一组数据的倒数得到
        def reciprocal(train, test, cols):
            for f in list(cols):
                if self.filter_feature == True:
                    if '1/'+ f  not in self.importance_f:
                        continue
                train['1/'+f] = 1.0 / train[f]
                test['1/' +f] = 1.0 / test[f]
            return train, test

        #将每一组数据除均值得到
        def divide_mean(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*1/mean'  not in self.importance_f:
                        continue
                train[f + '*1/mean' ] = train[f] / dataset[f].mean()
                test[f + '*1/mean'] = test[f] / dataset[f].mean()
            return train, test

        # 将每一组数据除极差得到
        def divide_max_sub_min(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*1/max_sub_min' not in self.importance_f:
                        continue
                train[f + '*1/max_sub_min'] = train[f] / (dataset[f].max()-dataset[f].min())
                test[f + '*1/max_sub_min'] = test[f] / (dataset[f].max()-dataset[f].min())
            return train, test

        # 将每一组数据除标准差得到
        def devide_std(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*1/variance' not in self.importance_f:
                        continue
                train[f + '*1/variance'] = train[f] / dataset[f].std()
                test[f + '*1/varianve'] = test[f] / dataset[f].std()
            return train, test

        # 将每一组数据乘均值得到
        def multiply_mean(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*mean' not in self.importance_f:
                        continue
                train[f + '*mean'] = train[f] * dataset[f].mean()
                test[f + '*mean'] = test[f] * dataset[f].mean()
            return train, test

        # 将每一组数据乘极差得到
        def multiply_max_sub_min(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*max_sub_min' not in self.importance_f:
                        continue
                train[f + '*max_sub_min'] = train[f] * (dataset[f].max() - dataset[f].min())
                test[f + '*max_sub_min'] = test[f] * (dataset[f].max() - dataset[f].min())
            return train, test

        # 将每一组数据乘标准差得到
        def multiply_std(train, test, cols):
            dataset = train.append(test)
            for f in list(cols):
                if self.filter_feature == True:
                    if f + '*variance' not in self.importance_f:
                        continue
                train[f + '*variance'] = train[f] * dataset[f].std()
                test[f + '*varianve'] = test[f] * dataset[f].std()
            return train, test

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
                train[f1 + '+' + f2 + name] = colx + coly

                colx = (test[f1] - dataset[f1].mean()) / dataset[f1].std()
                coly = (test[f2] - dataset[f2].mean()) / dataset[f2].std()
                test[f1 + '+' + f2 + name] = colx + coly
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

                colx = test[f1] / dataset[f1].std()
                coly = test[f2] / dataset[f2].std()
                test[f1 + '/' + f2] = colx / coly
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
                                             '性别'
                                             ])
        # contiCols = self.train.columns.drop(['性别'])
        print(len(self.train.columns))
        self.train, self.test = polynomial(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = addfeature(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = difffeature(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = divifeature(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = reciprocal(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = divide_max_sub_min(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = divide_mean(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = devide_std(self.train, self.test, contiCols)
        print(len(self.train.columns))
        self.train, self.test = multiply_mean(self.train, self.test, contiCols)
        print("乘均值：" + str(len(self.train.columns)))
        self.train, self.test = multiply_max_sub_min(self.train, self.test, contiCols)
        print("乘极差：" + str(len(self.train.columns)))
        self.train, self.test = multiply_std(self.train, self.test, contiCols)
        print("乘标准差：" + str(len(self.train.columns)))

    def category_sex(self):
        sex_dict = {"男": [1,0], "女": [0,1]}
        self.train = self.train.replace(sex_dict)
        self.test = self.test.replace(sex_dict)

    def translabelbelow(self, value, test=True):
        self.train_label = self.train_label.apply(lambda x: 1 if x <= value else 0)
        if test == False:
            self.test_label = self.test_label.apply(lambda x: 1 if x <= value else 0)

    def translabelup(self, value, test=True):
        self.train_label = self.train_label.apply(lambda x: 1 if x >= value else 0)
        if test == False:
            self.test_label = self.test_label.apply(lambda x: 1 if x >= value else 0)

    def drop_initial_columns(self):
        self.test = self.test.loc[:,self.importance_f]
        self.train = self.train.loc[:,self.importance_f]
