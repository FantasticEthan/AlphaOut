import pandas as pd
import numpy as np


class dataset(object):
    def __init__(self,path,train=True):

        self.data = pd.read_csv(path, sep=',', index_col='id')

        def preprocess(data):
            sex_dict = {"男": 1, "女": 0}
            data = data[data.columns.drop('体检日期')].replace(sex_dict)
            # describe = pd.read_csv("../tmp/describe.csv",index_col=[0])
            # fill the nan with dataframe.mean()
            data.fillna(round(data.mean(), 2), inplace=True)
            # reset
            data = data.reset_index(drop=True)
            num_data = len(data)
            return data
        self.preprocess_data = preprocess(self.data)

        def generate_matrix(data,train):
            if train ==1:
                X_matrix = data.iloc[:, :-1].values.astype(float)
                y_matrix = data.iloc[:, -1].values.astype(float)
            else:
                X_matrix = data.iloc[:, :].values.astype(float)
                y_matrix = np.array((len(data),1))
            return X_matrix,y_matrix

        if train==1:
            self.feature,self.label = generate_matrix(self.preprocess_data,train)
            self.index_in_epoch = 0
            self.example_nums = len(self.label)
            self.epochs_completed = 0
        else:
            self.feature,self.label = generate_matrix(self.preprocess_data,train)
            self.index_in_epoch = 0
            self.example_nums = len(self.label)
            self.epochs_completed = 0

    def normalization0_1(self):
        """
        :return: normalization data to (0,1)
        """
        self.norm_data = (self.preprocess_data - self.preprocess_data.min()) / (self.preprocess_data.max() - self.preprocess_data.min())
        self.norm_feature = self.norm_data.iloc[:, :-1].values.astype(float)
        self.norm_label = self.norm_data.iloc[:, -1].values.astype(float)
        return self.norm_feature,self.norm_label

    def Z_score(self):
        """
        :return: normalization to mean:0,variance:1
        """
        self.norm_feature = (self.feature - self.feature.mean()) / (self.feature.std())
        self.norm_label = (self.label - self.label.mean()) / (self.feature.std())
        return self.norm_feature, self.norm_label

    def next_batch(self,batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.feature = self.feature[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
        end = self.index_in_epoch
        return np.array(self.feature[start:end]), np.array(self.label[start:end])

