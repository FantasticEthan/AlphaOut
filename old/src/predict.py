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

modelpath = "../model/[(8, 2), (1.0, 0.8), 4, 0.01].model"

# data = data_helper.dataset("../tmp/train_all.csv",train=1)
# data = data_helper.dataset("../tmp/localtest.csv",train=False)
dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/onlinetest.csv",trainable=False)
dataset.trans_datetime2weather()
dataset.category_sex()
dataset.fillna_outliermean()

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