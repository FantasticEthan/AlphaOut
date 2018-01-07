import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import dataAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


train_data = pd.read_csv("../tmp/train.csv",sep=',',index_col='id')
test_data = pd.read_csv("../tmp/dev.csv",sep=',',index_col='id')

sex_dict = {"男":1,"女":0}
train_data = train_data[train_data.columns.drop('体检日期')].replace(sex_dict)
test_data = test_data[test_data.columns.drop('体检日期')].replace(sex_dict)

# describe = pd.read_csv("../tmp/describe.csv",index_col=[0])

train_data.fillna(round(train_data.mean(),2),inplace=True)
test_data.fillna(round(test_data.mean(),2),inplace=True)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

num_data = len(train_data)

X_train = train_data.iloc[:, :-1].values.astype(float)
y_train = train_data.iloc[:, -1].values.astype(float)

X_test = test_data.iloc[:, :-1].values.astype(float)
y_test = test_data.iloc[:, -1].values.astype(float)


poly_reg = PolynomialFeatures(degree = 2) #degree 就是自变量需要的维度
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
print("finish!")

train_predict = lin_reg_2.predict(poly_reg.fit_transform(X_train))
train_mseError = sum((train_predict - y_train)**2)/(2*num_data)

test_predict = lin_reg_2.predict(poly_reg.fit_transform(X_test))
test_mseError = sum((test_predict - y_test)**2)/(2*len(X_test))

print(("训练误差为{}..验证误差为{}..本地测试误差为...").format(train_mseError,test_mseError))
plt.scatter(range(num_data), y_train, color = 'red')
plt.plot(range(num_data), lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

