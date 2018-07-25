import pandas as pd
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge
from sklearn import metrics
from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA

filename = '../tmp/combineModel.csv'
test_filename = '../tmp/combineModel_onlinetest.csv'
data = pd.read_csv(filename,index_col='id')
data = data.fillna(data.mean())
test_data = pd.read_csv(test_filename,index_col='id')
test_data = test_data.fillna(data.mean())
X_test =test_data.iloc[:,:]

X_train = data.iloc[:,0:-1]
y_train = data.iloc[:,-1]

linreg = BayesianRidge()
linreg.fit(X_train, y_train)

print (linreg.intercept_)
print (linreg.coef_)

y_pred = linreg.predict(X_test)

dfpred = pd.DataFrame(y_pred,columns=["血糖"])
print(dfpred.describe())
dfpred.to_csv("submission.csv",header=None,index=None)
exit()
y_pred_train = linreg.predict(X_train)
print ("MSE:",metrics.mean_squared_error(y_train, y_pred_train))
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
# print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

