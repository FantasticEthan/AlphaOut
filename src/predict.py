import xgboost as xgb
import pandas as pd
import data_helper
import matplotlib.pyplot as plt
from operator import itemgetter


modelpath = "../model/21_cv5/dart.model"


dataset = data_helper.dataset("../tmp/train.csv","../tmp/onlinetest.csv",test = 1)
dataset.trans_datetime2weather()
#dataset.del_outlier()
dataset.fill_nan()
dataset.generate_arithmetic()
dataset.category_sex()
# dataset.whiteprotein_divd_creatinine()
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
# ceate_feature_map(dataset.train.columns)
X_train,y_train = dataset.train.values,dataset.train_label.values
# X_test,y_test = dataset.test.values,dataset.test_label.values
X_test= dataset.test.values

data_matrix = xgb.DMatrix(X_test)

print("predict...")
bst = xgb.Booster()
bst.load_model(modelpath)

# importance = bst.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(),key=itemgetter(1),reverse=True)
# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
# df['fscore'] = df['fscore'] / df['fscore'].sum()
# print(df[df['fscore']>0.003]['feature'].tolist())
# plt.figure()
# df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.show()

ypred = bst.predict(data_matrix)
dfpred = pd.DataFrame(ypred,columns=["血糖"])
print(dfpred.describe())
dfpred.plot()
plt.show()
# dfpred.to_csv("submission_18.csv",header=None,index=None)


def evalerror(preds, labels):
   error = sum((preds-labels)**2)/(2*len(preds))
   # return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
   return 'mse', error

# error = evalerror(ypred,y_test)
# print(error)






