# -*- coding: utf-8 -*-
import os
import xgboost as xgb
import pandas as pd
import data_helper
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.externals import joblib


xgbmodelpath = "../model/23_check_initial/xgb.model"
dartmodelpath = "../model/23_check_initial/dart.model"
rfmodelpath = "../model/23_check_initial/rf.pkl"

dataset = data_helper.dataset("../tmp/train.csv","../tmp/onlinetest.csv",test = 1)
dataset.trans_datetime2weather()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()

X_train,y_train = dataset.train.values,dataset.train_label.values
X_test= dataset.test.values

data_matrix = xgb.DMatrix(X_test)

print("predict  xgb ...")
bst = xgb.Booster()
bst.load_model(xgbmodelpath)
xgb_pred = bst.predict(data_matrix)

print("predict  dart ...")
bst = xgb.Booster()
bst.load_model(dartmodelpath)
dart_pred = bst.predict(data_matrix)

print("predict  rf ...")
clf = joblib.load(rfmodelpath)
rf_pred = clf.predict(X_test)


multi_pred = (xgb_pred*0.4+dart_pred*0.25+rf_pred*0.35)

dfpred = pd.DataFrame(multi_pred,columns=["血糖"])
print(dfpred.describe())
dfpred.plot()
plt.show()
dfpred.to_csv("submission_23_end.csv",header=None,index=None)



