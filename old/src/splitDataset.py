import pandas as pd
from dataAnalysis import load_dataset
from sklearn.cross_validation import train_test_split

filepath = "../data/d_train_20180102.csv"
tmppath  = "../tmp/"
data = load_dataset(filepath)

data.to_csv(tmppath+'train.csv')
exit()
datalength= len(data)
df_male = data[data["性别"]=="男"]
df_female = data[data["性别"]=="女"]
num_male = len(df_male)
num_female = len(df_female)
print(("男性测试者有{}个，女性测试者有{}个").format(num_male,num_female))

train_data = data.ix[:,:-1]
train_target = data.ix[:,["血糖"]]
# print(train_data)
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.15, random_state=30)

(X_train.join(y_train)).to_csv(tmppath+'train_all.csv')
X_train,X_Dev, y_train,y_Dev = train_test_split(X_train,y_train,test_size=0.1, random_state=27)

X_train_male = X_train[X_train["性别"]=="男"]
X_train_female = X_train[X_train["性别"]=="女"]

X_Dev_male = X_Dev[X_Dev["性别"]=="男"]
X_Dev_female = X_Dev[X_Dev["性别"]=="女"]

X_test_male = X_test[X_test["性别"]=="男"]
X_test_female = X_test[X_test["性别"]=="女"]

print(("训练集男性测试者有{}个，女性测试者有{}个").format(len(X_train_male),len(X_train_female)))
print(("验证集男性测试者有{}个，女性测试者有{}个").format(len(X_Dev_male),len(X_Dev_female)))
print(("测试集男性测试者有{}个，女性测试者有{}个").format(len(X_test_male),len(X_test_female)))

# exit()
# (X_train.join(y_train)).to_csv(tmppath+'train.csv')
(X_Dev.join(y_Dev)).to_csv(tmppath+'dev.csv')
(X_test.join(y_test)).to_csv(tmppath+'localtest.csv')


# print(len(X_train),len(X_Dev),len(X_test))

