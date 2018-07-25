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
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()


X_train,y_train = dataset.train.values,dataset.train_label.values
X_test,y_test = dataset.test.values,dataset.test_label.values
#split test and train

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params={'booster':'dart',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':6,
	'lambda':2,
	'subsample':0.4,
	'colsample_bytree':0.7,
	'min_child_weight':8,#5~10
	'eta': 0.005,
	'sample_type':'uniform',
	'normalize':'tree',
	'rate_drop':0.1,
	'skip_drop':0.9,
    'silent':1,
	'seed':87,
	'nthread':12,
	'gpu_id': 0,
    'max_bin':16,
    'tree_method':'gpu_hist'
	}

watchlist  = [(dtrain,'train'),(dtest,'test')]

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

watchlist  = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params,
                  dtrain,
                  feval=evalerror,
                  num_boost_round=bst_nb+50,
                  evals=watchlist)

model.save_model("../model/"+'23_check_initial/'+'dart'+".model")

