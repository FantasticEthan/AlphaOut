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

dataset = data_helper.dataset("../tmp/train.csv","../tmp/onlinetest.csv",test=1)
dataset.trans_datetime2weather()
#dataset.del_outlier()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()
value = 8
dataset.translabelbelow(value=value)

X_train,y_train = dataset.train.values,dataset.train_label.values
X_test = dataset.test.values

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test)

params={'booster':'gbtree',
	'objective': 'binary:logistic',
	'scale_pos_weight':float(len(X_train)-sum(y_train))/sum(y_train),
	'eval_metric': 'auc',
	'max_depth':6,
	'lambda':0,
	'subsample':0.65,
	'colsample_bytree':0.65,
	'eta': 0.002,
    'silent': 1,
	'seed':1024,
	'nthread':12
	}

watchlist  = [(dtrain,'train')]

#通过cv找最佳的nround
cv_log = xgb.cv(params,
                dtrain,
                num_boost_round=25000,
                nfold=10,
                metrics='auc',
                early_stopping_rounds=50,
                seed=1024)

bst_auc= cv_log['test-auc-mean'].max()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
bst_nb = cv_log.nb.to_dict()[bst_auc]
#train
watchlist  = [(dtrain,'train')]
model = xgb.train(params,
                  dtrain,
                  num_boost_round=bst_nb+50,
                  evals=watchlist)

model.save_model("../model/"+"classfication"+str(value)+".model")

#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame()
test_result["lt8_prob"] = test_y
# test_result["lt8_prob"] = test_result["lt8_prob"].apply(lambda x: 1 if x >= 0.5 else 0)
test_result.to_csv("../tmp/lt8_prob.csv")

print (bst_nb,bst_auc)
