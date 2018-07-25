import lightgbm as lgb
import numpy as np
import data_helper
from sklearn.cross_validation import train_test_split

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    error = sum((preds-labels)**2)/(2*len(preds))
    # return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
    return 'mse', error,False

dataset = data_helper.dataset("../tmp/train.csv","../tmp/localtest.csv",trainable=1)

dataset.trans_datetime2weather()
dataset.category_sex()
dataset.fillna_outliermean()
X_train,Y_train = dataset.train,dataset.train_label

N= 50
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0164  # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'  # or 'mae'
params['sub_feature'] = 0.58  # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.7  # sub_row
params['bagging_freq'] = 2
params['num_leaves'] = 12  # num_leaf
params['min_data'] = 10  # min_data_in_leaf
params['min_hessian'] = 0.01  # min_sum_hessian_in_leaf
params['verbose'] = 0
num_boost_round = 2000


select_params = []
#树结构 调参过程
gridsearch_params = [
    (num_leaves, min_sum_hessian_in_leaf)
    for num_leaves in range(100, 127)
    for min_sum_hessian_in_leaf in [1e-3,1e-2,1]
    ]

# Define initial best params and MAE
# min_mae = float("Inf")
min_mse = float("Inf")
best_params = None
for num_leaves, min_sum_hessian_in_leaf in gridsearch_params:
    print("CV with num_leaves={}, min_sum_hessian_in_leaf={}".format(
        num_leaves,
        min_sum_hessian_in_leaf))

    # Update our parameters
    params['max_depth'] = num_leaves
    params['min_child_weight'] = min_sum_hessian_in_leaf
    error = []
    for i in range(N):
        print("seed {} is training".format(i))
        x_train, x_test, y_train, y_test = train_test_split(
            X_train, Y_train, test_size=0.1, random_state=i + 1)

        d_train = lgb.Dataset(x_train, label=y_train)
        dtest = lgb.Dataset(x_test, label=y_test)

        clf = lgb.train(params, d_train,
                        num_boost_round=num_boost_round,
                        valid_sets=dtest,
                        valid_names = 'dtest',
                        feval=evalerror,
                        early_stopping_rounds=50,
                        verbose_eval=False
                        )

        clf.reset_parameter({"num_threads": 1})
        p_test = clf.predict(x_test)

        error1 = sum((y_test - p_test) ** 2)/(2*len(p_test))
        error2=  sum((y_test - y_train.mean()) ** 2)/(2*len(p_test))
        error.append(error1)

    mean_mse = np.mean(error)
    boost_rounds = 1
    print("\tmse {} for {} rounds".format(mean_mse, boost_rounds))
    if mean_mse < min_mse:
        min_mse = mean_mse
        best_params = (num_leaves, min_sum_hessian_in_leaf)
    # print(np.mean(error))
# print(sum1 / len(y_test) / N / 2)
# print(sum2 / len(y_test) / N / 2)
# print()