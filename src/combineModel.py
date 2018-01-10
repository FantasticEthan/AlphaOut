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

dataset = data_helper.dataset("../tmp/train_all.csv","../tmp/localtest.csv",trainable=1)
dataset.trans_datetime2weather()
dataset.del_outlier()

liver_train,liver_train_label, liver_test, liver_test_label = dataset.liver_columns()
bloodfat_train,bloodfat_train_label,bloodfat_test,bloodfat_test_label= dataset.bloodfat_columns()
urea_train,urea_train_label,urea_test,urea_test_label= dataset.urea_columns()
hepatitis_train,hepatitis_train_label,hepatitis_test,hepatitis_test_label= dataset.hepatitis()
bloodnorm_train,bloodnorm_train_label,bloodnorm_test,bloodnorm_test_label= dataset.bloodnorm()

typename = ['liver','bloodfat','urea','hepatitis','bloodnorm']


def combine_model(a,b,c,d,e,label_a,label_b,label_c,label_d,label_e):
    #1 predict liver
    data_matrix = xgb.DMatrix(a)
    print("predict...")
    bst = xgb.Booster()
    bst.load_model(liver_modelpath)
    ypred = bst.predict(data_matrix)
    label_a['liver'] = ypred
    print(label_a.describe())
    del bst

    #2 predict bloodfat
    data_matrix = xgb.DMatrix(b)
    print("predict...")
    bst = xgb.Booster()
    bst.load_model(bloodfat_modelpath)
    ypred = bst.predict(data_matrix)
    label_b['bloodfat'] = ypred
    print(label_b.describe())
    del bst

    #3 predict urea
    data_matrix = xgb.DMatrix(c)
    print("predict...")
    bst = xgb.Booster()
    bst.load_model(urea_modelpath)
    ypred = bst.predict(data_matrix)
    label_c['urea'] = ypred
    print(label_c.describe())
    del bst

    #4 predict hepatitis
    data_matrix = xgb.DMatrix(d)
    print("predict...")
    bst = xgb.Booster()
    bst.load_model(hepatitis_modelpath)
    ypred = bst.predict(data_matrix)
    label_d['hepatitis'] = ypred
    print(label_d.describe())
    del bst

    #5 predict bloodnorm
    data_matrix = xgb.DMatrix(e)
    print("predict...")
    bst = xgb.Booster()
    bst.load_model(bloodnorm_modelpath)
    ypred = bst.predict(data_matrix)
    label_e['bloodnorm'] = ypred
    print(label_e.describe())
    del bst

    df_label = pd.concat([label_a,label_b,
                          label_c,label_d,
                          label_e],axis=1)

    return df_label

#train data
label_a = pd.DataFrame(index=liver_train_label.index)
label_b = pd.DataFrame(index=bloodfat_train_label.index)
label_c = pd.DataFrame(index=urea_train_label.index)
label_d = pd.DataFrame(index=hepatitis_train_label.index)
label_e = pd.DataFrame(index=bloodnorm_train_label.index)

df_combine = combine_model(liver_train,bloodfat_train,urea_train,hepatitis_train,bloodnorm_train,
                           label_a,label_b,label_c,label_d,label_e)

df_combine = (df_combine.join(dataset.train_label))

df_combine.to_csv("../tmp/combineModel.csv")

#test data
label_a_test = pd.DataFrame(index=liver_test_label.index)
label_b_test = pd.DataFrame(index=bloodfat_test_label.index)
label_c_test = pd.DataFrame(index=urea_test_label.index)
label_d_test = pd.DataFrame(index=hepatitis_test_label.index)
label_e_test = pd.DataFrame(index=bloodnorm_test_label.index)

df_combine_test = combine_model(liver_test,bloodfat_test,urea_test,hepatitis_test,bloodnorm_test,
                           label_a_test,label_b_test,label_c_test,label_d_test,label_e_test)

df_combine_test = (df_combine_test.join(dataset.test_label))

df_combine_test.to_csv("../tmp/combineModel_test.csv")
# print(liver_test_label)
# liver_test_label.plot()
# plt.show()
# liver_test_label.to_csv("submission.csv",header=None,index=None)