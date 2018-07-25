import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_helper
# import dataAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

train_data = data_helper.dataset("../tmp/train.csv")
dev_data = data_helper.dataset("../tmp/dev.csv")
test_data = data_helper.dataset("../tmp/localtest.csv")
# y_train_temp = train_data.label

X_train,y_train = train_data.feature,train_data.label
X_dev,y_dev = dev_data.feature,dev_data.label
X_test,y_test = test_data.feature,test_data.label

num_data = train_data.example_nums

poly_reg = PolynomialFeatures(degree = 2) #degree 就是自变量需要的维度
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
print("finish!")

train_predict = lin_reg_2.predict(poly_reg.fit_transform(X_train))
train_mseError = sum((train_predict - y_train)**2)/(2*len(X_train))

dev_predict = lin_reg_2.predict(poly_reg.fit_transform(X_dev))
dev_mseError = sum((dev_predict - y_dev)**2)/(2*len(X_dev))

test_predict = lin_reg_2.predict(poly_reg.fit_transform(X_test))
test_mseError = sum((test_predict - y_test)**2)/(2*len(X_test))

print(("训练误差为{}..验证误差为{}..本地测试误差为{}...").format(train_mseError,dev_mseError,test_mseError))

plt.scatter(range(num_data), y_train, color = 'red')
plt.plot(range(num_data), lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


