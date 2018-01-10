import xgboost as xgb
import pandas as pd
import numpy as np
import data_helper
import operator
import matplotlib.pyplot as plt

modelpath = "../model/combine/"
liver_modelpath = "../model/combine/liver.model"
bloodfat_modelpath = "../model/combine/bloodfat.model"
urea_modelpath = "../model/combine/urea.model"
hepatitis_modelpath = "../model/combine/hepatitis.model"
bloodnorm_modelpath = "../model/combine/bloodnorm.model"

dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/onlinetest.csv",trainable=False)
dataset.trans_datetime2weather()
dataset.del_outlier()

_, _, liver_test, liver_test_label = dataset.liver_columns()
_,_,bloodfat_test,bloodfat_test_label= dataset.bloodfat_columns()
_,_,urea_test,urea_test_label= dataset.urea_columns()
_,_,hepatitis_test,hepatitis_test_label= dataset.hepatitis()
_,_,bloodnorm_test,bloodnorm_test_label= dataset.bloodnorm()
typename = ['liver','bloodfat','urea','hepatitis','bloodnorm']

#1 predict liver
data_matrix = xgb.DMatrix(liver_test)
print("predict...")
bst = xgb.Booster()
bst.load_model(liver_modelpath)
ypred = bst.predict(data_matrix)
liver_test_label['liver'] = ypred
print(liver_test_label.describe())
del bst

#2 predict bloodfat
data_matrix = xgb.DMatrix(bloodfat_test)
print("predict...")
bst = xgb.Booster()
bst.load_model(bloodfat_modelpath)
ypred = bst.predict(data_matrix)
bloodfat_test_label['bloodfat'] = ypred
print(bloodfat_test_label.describe())
del bst

#3 predict urea
data_matrix = xgb.DMatrix(urea_test)
print("predict...")
bst = xgb.Booster()
bst.load_model(urea_modelpath)
ypred = bst.predict(data_matrix)
urea_test_label['urea'] = ypred
print(urea_test_label.describe())
del bst

#4 predict hepatitis
data_matrix = xgb.DMatrix(hepatitis_test)
print("predict...")
bst = xgb.Booster()
bst.load_model(hepatitis_modelpath)
ypred = bst.predict(data_matrix)
hepatitis_test_label['hepatitis'] = ypred
print(hepatitis_test_label.describe())
del bst

#5 predict bloodnorm
data_matrix = xgb.DMatrix(bloodnorm_test)
print("predict...")
bst = xgb.Booster()
bst.load_model(bloodnorm_modelpath)
ypred = bst.predict(data_matrix)
bloodnorm_test_label['bloodnorm'] = ypred
print(bloodnorm_test_label.describe())
del bst

df_label = pd.concat([liver_test_label,bloodfat_test_label,
                      urea_test_label,hepatitis_test_label,
                      bloodnorm_test_label],axis=1)
print(df_label)

# print(liver_test_label)
# liver_test_label.plot()
# plt.show()
# liver_test_label.to_csv("submission.csv",header=None,index=None)