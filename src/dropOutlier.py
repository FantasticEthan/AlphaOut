import pandas as pd
import data_helper
import matplotlib.pyplot as plt

train_data = data_helper.dataset("../tmp/train_all.csv")
data = train_data.data
data = data.reset_index(drop=True)

data = data[data.columns.drop(["性别",'年龄',"体检日期"])]

std_outlier = data[(data-data.mean())>3*data.std()]

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1
outlier_step = 1.5 * IQR
box_outlier = data[(data < Q1 - outlier_step) | (data > Q3 + outlier_step)]

percent_outlier = data[(data>data.quantile(0.025)) | (data>data.quantile(0.975))]

# outlier_dict= {'a':std_outlier,"b":box_outlier,"c":percent_outlier}

outlier = data[((box_outlier.notnull()) & (std_outlier.notnull()))
           | ((box_outlier.notnull()) & (percent_outlier.notnull()))
           | ((std_outlier.notnull()) & (percent_outlier.notnull()))
           | ((box_outlier.notnull()) & (percent_outlier.notnull()) & (std_outlier.notnull()) )
      ]

print((outlier[outlier['*天门冬氨酸氨基转换酶'].notnull()]))
# print(outlier[outlier.notnull()])
# print(std_outlier.isnull())
# for i in data.columns:
#     if data[[i]]


# outlier_row.to_csv("../tmp/boxplot.csv")