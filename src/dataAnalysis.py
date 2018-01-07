import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class dataset(object):
    def __init__(self,feature,label):
        self.index_in_epoch = 0
        self.feature = feature
        self.label = label
        self.example_nums = len(label)
        self.epochs_completed = 0

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



def load_dataset(path):
    """
    string  :param path: the path to load
    dataframe  :return: dataframe of full data
    """
    df = pd.read_csv(path,index_col='id',encoding='gb2312')
    return df

def autolabel(ax,rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

def create_Histgram(dataframe,imgpath='../img/'):
    """
    create histgram for each feather except string and datetime type
    :param dataframe:
    :param imgpath:the path to sava
    :return:
    """
    columns = dataframe.columns.drop(['性别','体检日期'])
    for i in columns:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        dataframe[i].plot(kind='hist',grid=False,ax=ax1)
        autolabel(ax=ax1,rects=ax1.patches)
        plt.title(str(i))
        # plt.show()
        plt.savefig(imgpath+str(i)+"---histgram.png")
        plt.close()

def get_numerical_columns(data,tmp_path="../tmp/"):
    """

    :param data:dataframe
    :return:
    """
    data_numerical = data[data.columns.drop(['性别', '体检日期'])]

    df_describe = data_numerical.describe()
    df_describe.ix['num_missing'] = len(data_numerical) - df_describe.ix['count']
    # print(df_describe)
    df_describe.to_csv(tmp_path+"describe.csv")
    return data_numerical

def get_corr(data,tmp_path="../tmp/"):
    """

    :param data: origal dataframe
    :param tmp_path: temp folder to save file
    :return:correlation
    """
    sex_dict = {"男":1,"女":0}
    dftemp = data[data.columns.drop('体检日期')].replace(sex_dict)
    data_corr = dftemp.corr()
    data_corr_blood_glucose = dftemp.corr()['血糖']
    data_corr_blood_glucose.to_csv(tmp_path+"data_corr_blood_glucose.csv")
    return data_corr,data_corr_blood_glucose

if __name__=='__main__':
    filepath = "../data/d_train_20180102.csv"
    data = load_dataset(filepath)
    datalength= len(data)
    data = data.reset_index(drop=True)
    print(("总共有 {} 条 数据").format(datalength))
    # create_Histgram(data)
    data_numerical = get_numerical_columns(data)
    data_corr,data_corr_blood_glucose = get_corr(data)


