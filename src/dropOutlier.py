import pandas as pd
import data_helper
import numpy as np
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

# shape_ratio = (outlier.mean()-data.mean())/(outlier.std()-data.std()).abs()
Q1_outlier = outlier.quantile(0.25)
Q3_outlier = outlier.quantile(0.75)
IQR_outlier = Q3_outlier - Q1_outlier
outlier_outlier_step = 1.5 * IQR_outlier

box_outlier_outlier = outlier[(outlier < Q1_outlier - outlier_outlier_step) |
                      (outlier > Q3_outlier + outlier_outlier_step)]

outlier_baseline = box_outlier_outlier.min()

for i in data.columns:
    change = lambda x: outlier_baseline[i] if x>outlier_baseline[i] and x!='NaN' else x
    data[i] = data[i].map(change)


outlier_baseline.to_csv("../tmp/outlier_baseline.csv")

# print(data[data==float('inf')])
exit()

for i in box_outlier_outlier.columns:
    print(i)
    print(len(box_outlier_outlier[box_outlier_outlier[i].notnull()]))
exit()


df_importance_describe = pd.DataFrame()
for i in outlier.columns:
    dftemp = ((outlier[outlier[i].notnull()]['血糖']).describe())
    print(i)
    df_importance_describe[i] = dftemp

df_importance_describe.to_csv("../tmp/blood_value_dfsick_all.csv")
    # print(df_importance_describe)
# dfsick.to_csv("../tmp/dfsick_all.csv")


# outlier_row.to_csv("../tmp/boxplot.csv")