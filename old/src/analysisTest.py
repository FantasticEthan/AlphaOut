import pandas as pd
import data_helper

trainPath = "../tmp/train_all.csv"
localTestPath = "../tmp/localtest.csv"
onlineTestPath = "../data/d_train_20180102.csv"
addPath = "../tmp/createData2.csv"

# df_add = pd.read_csv(addPath,encoding='gb2312',index_col='id')
#
# df_train = pd.read_csv(trainPath,index_col='id')
#
# print(df_train)
# dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/onlinetest.csv",trainable=0)
#
# dataset.trans_datetime2weather()
# dataset.category_sex()
# dataset.fillna_outliermean()

localTest = pd.read_csv(localTestPath,sep=',',index_col='id')
# print(dataset.test.describe())
print(localTest.describe())


