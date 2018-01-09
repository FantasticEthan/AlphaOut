import pandas as pd
import data_helper
import matplotlib.pyplot as plt

train_data = data_helper.dataset("../tmp/onlinetest.csv")
data = train_data.data
data = data.reset_index(drop=True)
data["体检日期"] = pd.to_datetime(data["体检日期"])

# data =data.sort_values(by='体检日期',ascending=True)
num_data = len(data)
weatherpath = "../tmp/weather.csv"
weatherdata = pd.read_csv(weatherpath,index_col='日期',parse_dates=True)

data.insert(1,'high_temperature',weatherdata.ix[data['体检日期'],:]['最高'].tolist())
data.insert(1,'low_temperature',weatherdata.ix[data['体检日期'],:]['最低'].tolist())

bins = [0,7,18,41,66,100]
group_periods = [0,1,2,3,4]
cats = pd.cut(data.年龄, bins, right=False, labels=group_periods)
cats.rename('age_type')
data.insert(1,'age_type',cats.tolist())
# print(data)
data.to_csv("../tmp/addFeature1_onlinetest.csv",index_label='id')

exit()
data = data[data.columns.drop(["性别","age_type","low_temperature","high_temperature",'年龄',"体检日期"])]

gan = data[(data['*天门冬氨酸氨基转换酶'].isnull()==False)&(data['*丙氨酸氨基转换酶'].isnull()==False)&(data['*碱性磷酸酶'].isnull()==False)&(data['*r-谷氨酰基转换酶'].isnull()==False)&(data['*总蛋白'].isnull()==False)&(data['白蛋白'].isnull()==False)&(data['*球蛋白'].isnull()==False)&(data['白球比例'].isnull()==False)]
gan.to_csv("../tmp/gan.csv")
xuezhi = data[(data['甘油三酯'].isnull()==False)&(data['总胆固醇'].isnull()==False)&(data['高密度脂蛋白胆固醇'].isnull()==False)&(data['低密度脂蛋白胆固醇'].isnull()==False)]
xuezhi.to_csv("../tmp/xuezhi.csv")
niaosu = data[(data['尿素'].isnull()==False)&(data['肌酐'].isnull()==False)&(data['尿酸'].isnull()==False)]
niaosu.to_csv("../tmp/niaosu.csv")
yigan = data[(data['乙肝表面抗原'].isnull()==False)&(data['乙肝表面抗体'].isnull()==False)&(data['乙肝e抗原'].isnull()==False)&(data['乙肝e抗体'].isnull()==False)&(data['乙肝核心抗体'].isnull()==False)]
yigan.to_csv("../tmp/yigan.csv")
xuechanggui = data[(data['白细胞计数'].isnull()==False)&(data['红细胞计数'].isnull()==False)&(data['血红蛋白'].isnull()==False)&(data['红细胞压积'].isnull()==False)&(data['红细胞平均体积'].isnull()==False)&(data['红细胞平均血红蛋白量'].isnull()==False)&(data['红细胞平均血红蛋白浓度'].isnull()==False)&(data['红细胞体积分布宽度'].isnull()==False)&(data['血小板计数'].isnull()==False)&(data['血小板平均体积'].isnull()==False)&(data['血小板体积分布宽度'].isnull()==False)&(data['血小板比积'].isnull()==False)&(data['中性粒细胞%'].isnull()==False)&(data['淋巴细胞%'].isnull()==False)&(data['单核细胞%'].isnull()==False)&(data['嗜酸细胞%'].isnull()==False)&(data['嗜碱细胞%'].isnull()==False)]
xuechanggui.to_csv("../tmp/xuechanggui.csv")


# print(data[data["*天门冬氨酸氨基转换酶"].isnull()==False])
# sick_df = data[data['血糖']>11.1]
# sick_num = len(sick_df)
# sick_df.to_csv()