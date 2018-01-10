import xgboost as xgb
import pandas as pd
import numpy as np
import data_helper
import operator
import matplotlib.pyplot as plt

def ceate_feature_map(features):
    outfile = open('../tmp/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

modelpath = "../model/[(3, 5), (0.9, 0.7), 16,0.05].model"

# data = data_helper.dataset("../tmp/train_all.csv",train=1)
# data = data_helper.dataset("../tmp/localtest.csv",train=False)
dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/onlinetest.csv",train=False)
dataset.trans_datetime2weather()
dataset.del_outlier()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()

# train = dataset.train.values
# test = dataset.test.values

X_train,y_train = dataset.train.values,dataset.train_label.values
X_test = dataset.test.values

data_matrix = xgb.DMatrix(X_test)

ceate_feature_map(dataset.test.columns)
print("predict...")
bst = xgb.Booster()
bst.load_model(modelpath)
importance = bst.get_fscore(fmap='../tmp/xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)

ypred = bst.predict(data_matrix)

dfpred = pd.DataFrame(ypred,columns=["血糖"])
print(dfpred.describe())
dfpred.plot()
plt.show()
dfpred.to_csv("submission.csv",header=None,index=None)