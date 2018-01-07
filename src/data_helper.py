import pandas as pd
import numpy as np


class dataset(object):
    def __init__(self,path):

        self.data = pd.read_csv(path, sep=',', index_col='id')

        def generate_matrix(data):
            sex_dict = {"男": 1, "女": 0}
            data = data[data.columns.drop('体检日期')].replace(sex_dict)
            # describe = pd.read_csv("../tmp/describe.csv",index_col=[0])
            #fill the nan with dataframe.mean()
            data.fillna(round(data.mean(), 2), inplace=True)
            #reset
            data = data.reset_index(drop=True)
            num_data = len(data)

            X_matrix = data.iloc[:, :-1].values.astype(float)
            y_matrix = data.iloc[:, -1].values.astype(float)

            return X_matrix,y_matrix

        self.feature,self.label = generate_matrix(self.data)
        self.index_in_epoch = 0
        self.example_nums = len(self.label)
        self.epochs_completed = 0

    def normalization0_1(self):
        """
        :return: normalization data to (0,1)
        """
        self.norm_feature = (self.feature - self.feature.min()) / (self.feature.max() - self.feature.min())
        self.norm_label = (self.label - self.label.min()) / (self.label.max() - self.label.min())
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


