import xgboost as xgb
import pandas as pd
import numpy as np
import data_helper
import matplotlib.pyplot as plt

modelpath = "../model/[(4, 6), (0.9, 1.0), 0.01].model"

data = data_helper.dataset("../tmp/addFeature1_onlinetest.csv",train=False)
data_feature = data.feature

data_matrix = xgb.DMatrix(data_feature)

print("predict...")
bst = xgb.Booster()
bst.load_model(modelpath)
ypred = bst.predict(data_matrix)

dfpred = pd.DataFrame(ypred,columns=["血糖"])
print(dfpred.describe())
dfpred.plot()
plt.show()
dfpred.to_csv("submission.csv",header=None,index=None)