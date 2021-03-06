import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_helper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
import xgboost as xgb


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    error = sum((preds-labels)**2)/(2*len(preds))
    # return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
    return 'mse', error

dataset = data_helper.dataset("../tmp/train.csv","../tmp/localtest.csv")
dataset.trans_datetime2weather()
#dataset.del_outlier()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()

X_train,y_train = dataset.train.values,dataset.train_label.values
X_test,y_test = dataset.test.values,dataset.test_label.values
#split test and train

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#train part
params = {
    # Parameters that we are going to tune.
    'max_depth': 6,
    'min_child_weight': 8,
    'eta': 0.005,
    'subsample': 0.4,
    'colsample_bytree': 0.7,
    'lambda':0,
    # Other parameters
    'silent':1,
    'objective':'reg:linear',
    'gpu_id': 0,
    'max_bin':16,
    'tree_method':'gpu_hist'
}

#通过cv找最佳的nround
cv_log = xgb.cv(params,
                dtrain,
                num_boost_round=2000,
                nfold=5,
                feval=evalerror,
                early_stopping_rounds=50,
                seed=24)

bst_rmse= cv_log['test-rmse-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-rmse-mean']
bst_nb = cv_log.nb.to_dict()[bst_rmse]
#------------------------
#-----迭代次数-------------
num_boost_round = 2000

model = xgb.train(
    params,
    dtrain,
    num_boost_round=bst_nb+50,
    evals=[ (dtrain, 'train'),(dtest, 'eval')],
    feval=evalerror,
    early_stopping_rounds=50
)

print("Best mse: {:.2f} with {} rounds".format(
    model.best_score,
    model.best_iteration + 1))

# model.save_model("../model/"+'23_check_initial/'+'xgb'+".model")
#
#
# exit()






#交叉验证
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=9,
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
    for max_depth in range(3, 7)
    for min_child_weight in range(1, 8)
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
        seed=9,
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
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]
select_params.append(best_params)
#subsample colsample 调参过程
#——————————————————————————————————————————————
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i / 10. for i in range(3, 11)]
    for colsample in [i / 10. for i in range(3, 11)]
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
        seed=9,
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
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]
select_params.append(best_params)

#lambda 调参过程
# %
# This can take some time…
min_mse = float("Inf")
best_params = None

for reg_lambda in range(1,20):
    print("CV with reg_lambda={}".format(reg_lambda))

    # We update our parameters
    params['lambda'] = reg_lambda

    # Run and time CV
    # %time
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=9,
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
        best_params = reg_lambda

print("Best params: {}, mse: {}".format(best_params, min_mse))
params['lambda'] = best_params
select_params.append(best_params)

print(select_params)

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
        seed=9,
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
params['eta'] = best_params
select_params.append(best_params)

print(select_params)

max_depth_param,min_child_param = select_params[0][0],select_params[0][1]
subsample_param,colsample_param = select_params[1][0],select_params[1][1]
reg_lambda_param = select_params[2]
eta_param = select_params[3]

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








