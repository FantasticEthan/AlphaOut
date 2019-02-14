from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import pandas as pd
import numpy as np
import data_helper
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score

dataset = data_helper.dataset("../tmp/train.csv","../tmp/localtest.csv")
dataset.trans_datetime2weather()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()

model = RandomForestRegressor(n_estimators=1000,
                              criterion='mse',
                              max_depth=6,
                              max_features=0.8,
                              min_samples_leaf=8,
                              n_jobs=12,
                              random_state=777)#min_samples_leaf: 5~10

X_train,y_train = dataset.train.values,dataset.train_label.values
X_test,y_test = dataset.test.values,dataset.test_label.values

scores = cross_val_score(model,
                         X_train,
                         y_train,
                         cv=5,
                         scoring='mean_squared_error')

print (np.sqrt(-scores),np.mean(np.sqrt(-scores)))

model.fit(X_train,y_train)

joblib.dump(model, '../model/23_check_initial/rf.pkl')