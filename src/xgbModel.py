import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_helper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb

from xgboost import XGBClassifier
# from collections import Counter

def evalerror(preds, dtrain):
    labels = dtrain.get_label()

    error = sum((preds-labels)**2)/(2*len(preds))
    # return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
    return 'mse', error


train_data = data_helper.dataset("../tmp/addFeature1_train_all.csv")
test_data = data_helper.dataset("../tmp/addFeature1_localtest.csv")


X_train,y_train = train_data.feature,train_data.label
X_test,y_test = test_data.feature,test_data.label
#split test and train
X_matrix, y_matrix = X_train, y_train
testX_matrix, testy_matrix = X_test, y_test

dtrain = xgb.DMatrix(X_matrix, label=y_matrix)
dtest = xgb.DMatrix(testX_matrix, label=testy_matrix)

#train part
params = {
    # Parameters that we are going to tune.
    'max_depth': 11,
    'min_child_weight': 7,
    'eta': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    # Other parameters
    'silent':1,
    'objective':'reg:linear',
}

#------------------------
#-----迭代次数-------------
num_boost_round = 472

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, 'eval'), (dtrain, 'train')],
    feval=evalerror,
    early_stopping_rounds=50
)

print("Best mse: {:.2f} with {} rounds".format(
    model.best_score,
    model.best_iteration + 1))

#交叉验证
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    feval=evalerror,
    early_stopping_rounds=10
)
# print(cv_results)
# exit()
print("min is ...",cv_results['test-mse-mean'].min())

select_params = []
#树结构 调参过程
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(4, 12)
    for min_child_weight in range(5, 8)
    ]

# Define initial best params and MAE
# min_mae = float("Inf")
min_mse = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
        max_depth,
        min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        feval=evalerror,
        early_stopping_rounds=10
    )

    mean_mse = cv_results['test-mse-mean'].min()
    boost_rounds = cv_results['test-mse-mean'].argmin()
    print("\tmse {} for {} rounds".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = (max_depth, min_child_weight)

print("Best params: {}, {}, mse: {}".format(best_params[0], best_params[1], min_mse))

select_params.append(best_params)
#subsample colsample 调参过程
#——————————————————————————————————————————————
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i / 10. for i in range(7, 11)]
    for colsample in [i / 10. for i in range(7, 11)]
    ]
min_mse = float("Inf")
best_params = None

# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
        subsample,
        colsample))

    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        feval=evalerror,
        early_stopping_rounds=10
    )

    # Update best score
    mean_mse = cv_results['test-mse-mean'].min()
    boost_rounds = cv_results['test-mse-mean'].argmin()
    print("\tmse {} for {} rounds".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = (subsample, colsample)

print("Best params: {}, {}, mse: {}".format(best_params[0], best_params[1], min_mse))

select_params.append(best_params)
#Eta 调参过程
# %
# This can take some time…
min_mse = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))

    # We update our parameters
    params['eta'] = eta

    # Run and time CV
    # %time
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        feval=evalerror,
        early_stopping_rounds=10
    )

    # Update best score
    mean_mse = cv_results['test-mse-mean'].min()
    boost_rounds = cv_results['test-mse-mean'].argmin()
    print("\tmse {} for {} rounds\n".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = eta

print("Best params: {}, mse: {}".format(best_params, min_mse))

select_params.append(best_params)

print(select_params)

max_depth_param,min_child_param = select_params[0][0],select_params[0][1]
subsample_param,colsample_param = select_params[1][0],select_params[1][1]
eta_param = select_params[2]

params_best = {
    # Parameters that we are going to tune.
    'max_depth': max_depth_param,
    'min_child_weight': min_child_param,
    'eta': eta_param,
    'subsample': subsample_param,
    'colsample_bytree': colsample_param,
    # Other parameters
    'silent':1,
    'objective':'reg:linear',
}

model_best = xgb.train(
    params_best,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, 'eval'), (dtrain, 'train')],
    feval=evalerror,
    early_stopping_rounds=50
)

model_best.save_model("../model/"+str(select_params)+".model")