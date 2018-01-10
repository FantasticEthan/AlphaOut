import xgboost as xgb
import pandas as pd
import numpy as np
import data_helper
import matplotlib.pyplot as plt

modelpath = "../model/[(3, 9), (1.0, 0.9), 17, 0.01].model"

# data = data_helper.dataset("../tmp/train_all.csv",train=1)
# data = data_helper.dataset("../tmp/localtest.csv",train=False)
data = data_helper.dataset("../tmp/onlinetest.csv",train=False)

# print(data.data['血糖'].describe())

data_feature = data.data
data_mean = pd.read_csv("../tmp/train_mean.csv",names=['类别','数值'],index_col='类别')
baseline_outliers = pd.read_csv("../tmp/outlier_baseline.csv",names=['类别','数值'],index_col='类别')

# print(data_mean)
sex_dict = {"男": 1, "女": 0}
data_feature = data_feature.replace(sex_dict)
for i in data_feature.columns:
    data_feature[i] = data_feature[i].fillna(data_mean.ix[[i],0][0])

data_feature["体检日期"] = pd.to_datetime(data_feature["体检日期"])

num_data = len(data_feature)
weatherpath = "../tmp/weather.csv"
weatherdata = pd.read_csv(weatherpath, index_col='日期', parse_dates=True)

data_feature.insert(1, 'high_temperature', weatherdata.ix[data_feature['体检日期'], :]['最高'].tolist())
data_feature.insert(1, 'low_temperature', weatherdata.ix[data_feature['体检日期'], :]['最低'].tolist())

for i in data_feature.columns.drop(["性别", '年龄', "体检日期",'high_temperature','low_temperature']):
    baseline = baseline_outliers.ix[[i],0][0]
    change = lambda x: baseline if x>baseline else x
    data_feature[i] = data_feature[i].map(change)

data_feature = data_feature.reindex(columns =['*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白',
       '*球蛋白', '白球比例', '甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '尿素', '肌酐',
       '尿酸', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '白细胞计数', '红细胞计数',
       '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度',
       '血小板计数', '血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%',
       '嗜酸细胞%', '嗜碱细胞%', '性别', '年龄', 'high_temperature', 'low_temperature'])
# data_feature = data.data.iloc[:,:-1]

# print(data_feature)
# print(data_feature.columns,len(data_feature.columns))
data_matrix = xgb.DMatrix(data_feature)

print("predict...")
bst = xgb.Booster()
bst.load_model(modelpath)
# print(bst.get_fscore())
ypred = bst.predict(data_matrix)

dfpred = pd.DataFrame(ypred,columns=["血糖"])
print(dfpred.describe())
dfpred.plot()
plt.show()
dfpred.to_csv("submission.csv",header=None,index=None)