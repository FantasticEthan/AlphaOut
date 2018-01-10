import pandas as pd
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge
from sklearn import metrics
from itertools import combinations
import numpy as np
from sklearn.decomposition import PCA

filename = '../tmp/combineModel.csv'
test_filename = '../tmp/combineModel_test.csv'
data = pd.read_csv(filename,index_col='id').fillna(0)
test_data = pd.read_csv(test_filename,index_col='id').fillna(0)

X_test,y_test =test_data.iloc[:,0:-1],test_data.iloc[:,-1]
X_train = data.iloc[:,0:-1]
y_train = data.iloc[:,-1]

linreg = Ridge(alpha=0.5)
linreg.fit(X_train, y_train)

print (linreg.intercept_)
print (linreg.coef_)

y_pred = linreg.predict(X_test)

y_pred_train = linreg.predict(X_train)
print ("MSE:",metrics.mean_squared_error(y_train, y_pred_train))
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
# print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))