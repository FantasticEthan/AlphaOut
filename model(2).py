import pandas as pd
from sklearn.cluster import KMeans
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from hyperopt import fmin,tpe,hp,space_eval
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import xgboost as xgb
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from dateutil.parser import parse
import time
import datetime
from sklearn.cluster import  DBSCAN

def read_dataSet(trainfile='d_train_20180102.csv',testfile='d_test_A_20180102.csv'):
    usecol=[
        # 'id',
        '性别',
        '年龄',
        '体检日期',

        '*天门冬氨酸氨基转换酶',
        '*丙氨酸氨基转换酶',
        '*碱性磷酸酶',
        '*r-谷氨酰基转换酶',
        '*总蛋白',
        # '白蛋白',
        '*球蛋白',
        '白球比例',
        '甘油三酯',
        '总胆固醇',
        '高密度脂蛋白胆固醇',
        '低密度脂蛋白胆固醇',
        '尿素',
        '肌酐',
        '尿酸',

        # '乙肝表面抗原',
        # '乙肝表面抗体',
        # '乙肝e抗原',
        # '乙肝e抗体',
        # '乙肝核心抗体',

        '白细胞计数',
        '红细胞计数',
        '血红蛋白',
        '红细胞压积',
        '红细胞平均体积',
        '红细胞平均血红蛋白量',
        '红细胞平均血红蛋白浓度',
        '红细胞体积分布宽度',
        '血小板计数',
        '血小板平均体积',
        '血小板体积分布宽度',
        '血小板比积',
        '中性粒细胞%',
        '淋巴细胞%',
        '单核细胞%',
        # '嗜酸细胞%',
        # '嗜碱细胞%',
    ]
    # read_csv 读取CSV格式文件  usecols：读取哪几列   encoding：中文字符的编码处理
    # 这个返回了一个 DataFrame对象
    train = pd.read_csv(trainfile, usecols=usecol ,encoding='gb2312')
    test = pd.read_csv(testfile, usecols=usecol,encoding='gb2312')
    #  .values  DataFrame对象存储的(sample,features)矩阵
    label=pd.read_csv(trainfile,usecols=['血糖'] ,encoding='gb2312')['血糖'].values

    def deal_outlier(train):
        train.loc[train['*天门冬氨酸氨基转换酶']==434.95,'*天门冬氨酸氨基转换酶']=28.7375
        train.loc[train['*天门冬氨酸氨基转换酶']==158.87, '*天门冬氨酸氨基转换酶']=28.7375
        train.loc[train['*天门冬氨酸氨基转换酶']==155.52, '*天门冬氨酸氨基转换酶']=28.7375
        train.loc[train['*天门冬氨酸氨基转换酶']==135.23, '*天门冬氨酸氨基转换酶']=28.7375
        train.loc[train['*天门冬氨酸氨基转换酶']==146.64, '*天门冬氨酸氨基转换酶']=28.7375
        train.loc[train['*天门冬氨酸氨基转换酶']==281.21, '*天门冬氨酸氨基转换酶']=32.0575
        train.loc[train['*天门冬氨酸氨基转换酶']==121.62, '*天门冬氨酸氨基转换酶']=32.0575

        train.loc[train['*丙氨酸氨基转换酶'] ==498.89, '*丙氨酸氨基转换酶'] =30.9475
        train.loc[train['*丙氨酸氨基转换酶'] ==388, '*丙氨酸氨基转换酶'] = 37.5375
        train.loc[train['*丙氨酸氨基转换酶'] ==289.96, '*丙氨酸氨基转换酶'] =45.29
        train.loc[train['*丙氨酸氨基转换酶'] >200, '*丙氨酸氨基转换酶'] = 30.9475

        train.loc[train['*碱性磷酸酶'] > 300, '*碱性磷酸酶'] = 150
        train.loc[train['*碱性磷酸酶'] > 200, '*碱性磷酸酶'] = 200

        train.loc[train['*r-谷氨酰基转换酶'] > 250, '*r-谷氨酰基转换酶'] = 250

        train.loc[train['白球比例'] > 7, '白球比例'] = 2.47

        train.loc[train['甘油三酯'] > 15, '甘油三酯'] = 15

        train.loc[train['总胆固醇'] > 15, '总胆固醇'] = 11

        train.loc[train['高密度脂蛋白胆固醇'] >4, '高密度脂蛋白胆固醇'] =2.8

        train.loc[train['低密度脂蛋白胆固醇'] >8.5, '低密度脂蛋白胆固醇'] = 7

        train.loc[train['尿素'] >9, '尿素'] = 6.2

        train.loc[train['乙肝表面抗原'] > 2.4, '乙肝表面抗原'] = 2.14

        train.loc[train['乙肝e抗原'] > 2.4, '乙肝e抗原'] = 1

        train.loc[train['乙肝核心抗体'] > 11, '乙肝核心抗体'] = 10

        train.loc[train['白细胞计数'] > 15.7, '白细胞计数'] = 15

        train.loc[train['血小板计数'] > 550, '血小板计数'] = 550

        train.loc[train['血小板比积'] > 0.6, '血小板比积'] = 0.58

        train.loc[train['淋巴细胞%'] > 70, '淋巴细胞%'] = 62

        train.loc[train['单核细胞%'] > 22, '单核细胞%'] = 16

        train.loc[train['嗜酸细胞%'] >10, '嗜酸细胞%'] = 10

        train.loc[train['嗜碱细胞%'] > 2.5, '嗜碱细胞%'] = 2.3






        return train

    # train=deal_outlier(train)
    # test=deal_outlier(test)


    return train,test,label

def get_dummies(data, feature):
    #
    '''    get_dummies   传入Series类型的对象   返回一个one-hot编码过的DataFrame
            比如  传入['男','女','未知','未知','男']
            返回
            [[1,0,0],
             [0,1,0],
             [0,0,1],
             [0,0,1],
             [1,0,0]]
                返回的DataFrame列名前缀为prefix   名字整个为  feature_男  feature_女  feature_未知
    '''
    dummies = pd.get_dummies(data[feature], prefix=feature)
    '''
        concat 横向拼接DataFrame存储的矩阵
        axis是坐标轴
    '''
    data = pd.concat([data, dummies], axis=1)
    '''
        drop丢弃指定列（传个list指定）
        axis=0 则丢弃指定行
    '''
    return data.drop([feature], axis=1)

def deal_null(dataset):
    '.columns  是DataFrame的列名 list'
    columns = list(dataset.columns)
    columns.remove('性别')

    '''
        dataset[col]  DataFrame索引某列的方式  
        .mean() 该列的平均值   
        .fillna()  就是fill nan  填补缺失值
    '''
    for col in columns:
        dataset[col] = dataset[col].fillna(dataset[col].mean())
    return dataset

def deal_null_v2(train,test):
    dataset=train.append(test)
    columns = list(dataset.columns)
    columns.remove('性别')
    for col in columns:
        train[col] = train[col].fillna(dataset[col].mean())
        test[col] = test[col].fillna(dataset[col].mean())
    return train,test


class model:

    def __init__(self):
        print('load data')

        self.train,self.test,self.label=read_dataSet()
        self.train['体检日期'] = (pd.to_datetime(self.train['体检日期']) - parse('2017-08-09')).dt.days
        self.test['体检日期'] = (pd.to_datetime(self.test['体检日期']) - parse('2017-08-09')).dt.days
        weather=pd.read_csv('weather.csv',encoding='gb2312')
        weather['体检日期']=weather['date']
        weather.drop(['date'],axis=1,inplace=True)
        weather['体检日期']=(pd.to_datetime(weather['体检日期'])- parse('2017-08-09')).dt.days


        self.train=self.train.merge(weather, on=['体检日期'], how='left')
        self.test = self.test.merge(weather, on=['体检日期'], how='left')
        self.train['diff_tem']=self.train['maxvalue']-self.train['minvalue']
        self.test['diff_tem'] = self.test['maxvalue'] - self.test['minvalue']
        # self.train['ave_tem']=self.train['maxvalue']+self.train['minvalue']
        # self.test['ave_tem'] = self.test['maxvalue'] + self.test['minvalue']
        self.train.drop(['maxvalue','minvalue'],axis=1,inplace=True)
        self.test.drop(['maxvalue', 'minvalue'], axis=1, inplace=True)

        self.train,self.test=deal_null(self.train),deal_null(self.test)

        self.train, self.test =deal_null_v2(self.train,self.test)
        # self.PCA_dec()
        self.importance_f = [

            '年龄&血红蛋白',
            '甘油三酯-尿酸',
            '年龄+红细胞压积',
            '年龄&*r-谷氨酰基转换酶',
            '尿酸-红细胞计数',
            '年龄+红细胞平均血红蛋白浓度',
            '年龄+甘油三酯',
            '年龄+红细胞计数',
            '红细胞平均体积+红细胞体积分布宽度',
            '高密度脂蛋白胆固醇+尿酸',
            '尿酸-血红蛋白',

        ]





        "cols_kmeans 是拿那些特征来聚类"
        cols_kmeans=[
            '年龄',
            # '体检日期',
            '*天门冬氨酸氨基转换酶',
            '*丙氨酸氨基转换酶',
            '*碱性磷酸酶',
            '*r-谷氨酰基转换酶',
            '白球比例',
            '甘油三酯',
            '总胆固醇',
            '尿素',
            '尿酸',
            '红细胞计数',
            '血红蛋白',
            '红细胞压积',
            '红细胞平均血红蛋白量',
            '红细胞平均血红蛋白浓度',
            '红细胞体积分布宽度',
            # '血小板计数',
            # '血小板体积分布宽度',
            # '血小板比积',
            '中性粒细胞%',
            # '淋巴细胞%',
            # '单核细胞%',
            # '嗜碱细胞%',
            # 'diff_tem',
            # 'maxvalue',
            # 'minvalue',

        ]

        self.k_mean(cols_kmeans)



        # for col1,col2 in tqdm((combinations(cols_kmeans,2))):
        #     self.k_mean([col1,col2],col1+'_K_'+col2,4,iters=500)

        #

        "contiCols 是需要多项式处理的特征"
        contiCols = [
            '年龄',
            # 'maxvalue',
            # 'minvalue',
            # 'diff',
            '*天门冬氨酸氨基转换酶',
            '*丙氨酸氨基转换酶',
            '*碱性磷酸酶',
            '*r-谷氨酰基转换酶',
            '*总蛋白',
            # '白蛋白',
            '*球蛋白',
            '白球比例',
            '甘油三酯',
            '总胆固醇',
            '高密度脂蛋白胆固醇',
            '低密度脂蛋白胆固醇',
            '尿素',
            '肌酐',
            '尿酸',

            # '乙肝表面抗原',
            # '乙肝表面抗体',
            # '乙肝e抗原',
            # '乙肝e抗体',
            # '乙肝核心抗体',

            '白细胞计数',
            '红细胞计数',
            '血红蛋白',
            '红细胞压积',
            '红细胞平均体积',
            '红细胞平均血红蛋白量',
            '红细胞平均血红蛋白浓度',
            '红细胞体积分布宽度',

            '血小板计数',
            '血小板平均体积',
            '血小板体积分布宽度',
            # '血小板比积',

            '中性粒细胞%',
            '淋巴细胞%',
            '单核细胞%',
            # '嗜酸细胞%',
            # '嗜碱细胞%',
        ]

        self.filter_feature =True

        self.train, self.test = self.polynomial(self.train,self.test,contiCols)
        self.train, self.test = self.addfeature(self.train, self.test, contiCols)
        self.train, self.test = self.difffeature(self.train, self.test, contiCols)
        self.train, self.test = self.divifeature(self.train, self.test, contiCols)

        categoryCols=[
            '年龄',
            '体检日期',
            '*天门冬氨酸氨基转换酶',
            '*丙氨酸氨基转换酶',
            '*碱性磷酸酶',
            '*r-谷氨酰基转换酶',
            '白球比例',
            '甘油三酯',
            '总胆固醇',
            '尿素',

            '红细胞计数',
            '血红蛋白',
            '红细胞压积',
            '红细胞平均血红蛋白量',
            '红细胞平均血红蛋白浓度',
            '红细胞体积分布宽度',
            '血小板计数',
            '血小板体积分布宽度',
            '血小板比积',
            '中性粒细胞%',
            '淋巴细胞%',
            '单核细胞%',
            '嗜碱细胞%',

        ]
        # self.train,self.test=self.quantile_feature(self.train,self.test,categoryCols)



    def k_mean(self,cols,name='Kclass',k=5,iters=3000):
        dataset=self.train[cols].append(self.test[cols])
        dataset.index=range(len(self.train)+len(self.test))
        for col in cols:
            dataset[col]=(dataset[col]-dataset[col].mean())/dataset[col].std()

        model=KMeans(n_clusters=k,max_iter=iters,init='random')
        n_class=model.fit_predict(dataset.values)

        self.train[name]=n_class[:self.train.shape[0]]
        self.test[name]=n_class[self.train.shape[0]:]

        self.train[name]=self.train[name].astype('category')
        self.test[name] = self.test[name].astype('category')
        # self.train=get_dummies(self.train,'Kclass')
        # self.test =get_dummies(self.test, 'Kclass')

    def polynomial(self,train,test,cols):
        dataset=self.train.append(self.test)
        for f1,f2 in list(combinations(cols,2)):
            if self.filter_feature==True:
                if f1 + '&' + f2 not in self.importance_f:
                    continue
            colx=train[f1]/dataset[f1].std()
            coly =train[f2]/dataset[f2].std()
            train[f1+'&'+f2]=colx*coly

            colx =test[f1] / dataset[f1].std()
            coly =test[f2] / dataset[f2].std()
            test[f1 + '&' + f2] = colx * coly
        return train,test

    def difffeature(self,train,test,cols,w1=1,w2=1,name=''):
        dataset = self.train.append(self.test)
        for f1, f2 in list(combinations(cols, 2)):
            if self.filter_feature==True:
                if f1 + '-' + f2+name not in self.importance_f:
                    continue
            colx=(train[f1]-dataset[f1].mean())/dataset[f1].std()
            coly=(train[f2]-dataset[f2].mean())/dataset[f2].std()
            train[f1 + '-' + f2+name] = colx-coly

            colx = (test[f1] - dataset[f1].mean()) / dataset[f1].std()
            coly = (test[f2] - dataset[f2].mean()) / dataset[f2].std()
            test[f1 + '-' + f2+name] = colx-coly
        return train, test

    def addfeature(self,train,test,cols,w1=1,w2=1,name=''):
        dataset = self.train.append(self.test)
        for f1, f2 in list(combinations(cols, 2)):
            if self.filter_feature==True:
                if f1 + '+' + f2 +name not in self.importance_f:
                    continue
            colx=(train[f1]-dataset[f1].mean())/dataset[f1].std()
            coly=(train[f2]-dataset[f2].mean())/dataset[f2].std()
            train[f1 + '+' + f2+name] = w1*colx+w2*coly

            colx = (test[f1] - dataset[f1].mean()) / dataset[f1].std()
            coly = (test[f2] - dataset[f2].mean()) / dataset[f2].std()
            test[f1 + '+' + f2+name] = w1*colx + w2*coly
        return train, test

    def divifeature(self,train,test,cols):
        dataset = self.train.append(self.test)
        for f1, f2 in list(combinations(cols, 2)):
            if self.filter_feature==True:
                if f1 + '/' + f2 not in self.importance_f:
                    continue
            colx=train[f1]/dataset[f1].std()
            coly=train[f2]/dataset[f2].std()
            train[f1 + '/' + f2] = colx/coly
            # train[f2 + '/' + f1] = coly / colx

            colx = test[f1]  / dataset[f1].std()
            coly = test[f2] / dataset[f2].std()
            test[f1 + '/' + f2] = colx / coly
            # test[f2 + '/' + f1] = coly / colx
        return train, test

    def quantile_feature(self,train,test,cols,groups=2):

        dataset=train.append(test)
        for col in cols:
            dataset['q_'+col]=pd.qcut(dataset[col],groups,labels=False)

        for f1,f2 in list(combinations(cols,2)):
            dataset[f1+'_q_'+f2]=groups * dataset['q_'+f1]+dataset['q_'+f2]
            train[f1+'_q_'+f2]=dataset[f1+'_q_'+f2].values[:train.shape[0]]
            test[f1 + '_q_' + f2] = dataset[f1 + '_q_' + f2].values[train.shape[0]:]
            train[f1+'_q_'+f2]=train[f1+'_q_'+f2].astype('category')
            test[f1+'_q_'+f2]=test[f1+'_q_'+f2].astype('category')
        return train,test

    def PCA_dec(self,n_components=50):
        pca = PCA(n_components=n_components)
        dataset=self.train.append(self.test)
        pca.fit(dataset.values)
        self.train=pd.DataFrame(pca.transform(self.train.values),columns=list(range(n_components)))
        self.test=pd.DataFrame(pca.transform(self.test.values),columns=list(range(n_components)))

    def xgbmodel(self,istrain=False):

        def mse(preds, dtrain):
            labels = dtrain.get_label()
            return 'mse', float(sum((preds-labels)**2)) / (2*len(labels))

        def define_loss(preds, dtrain):

            pass
            # return grad, hess

        self.train = get_dummies(self.train, '性别')
        self.test  = get_dummies(self.test, '性别')

        X = self.train
        Y = self.label
        columns=list(X.columns)

        trainset=xgb.DMatrix(X.values,label=Y,feature_names=columns)

        watch_list = [(trainset,'train') ]
        param={
            'eta':0.013,
            'application':'regression',
            # 'eval_metric':mse,
            'colsample_bytree':0.37,
            'max_depth':12,
            'lambda':1,
            'min_child_weight':7,
            'silent':1,
        }

        if istrain==True:
            model=xgb.train(params=param,dtrain=trainset,num_boost_round=636,
                            # obj=define_loss,
                            feval=mse,
                            evals=watch_list,
                            early_stopping_rounds=100)
            score = model.get_score(importance_type='gain')

            score=pd.DataFrame(list(score.items()),columns=['feature','gain'])
            print(score.sort_values(by=['gain'], ascending=False))

            test = xgb.DMatrix(self.test, feature_names=columns)
            predict = model.predict(test)
            subm = pd.DataFrame(predict)
            subm.to_csv('submission.csv', index=False, header=False)

        else:
            train_preds = np.zeros(X.shape[0])
            test_preds = np.zeros((self.test.shape[0], 5))
            kf = KFold(len(X), n_folds=5, shuffle=True, random_state=520)

            X['血糖']=self.label

            xgb_test=xgb.DMatrix(self.test.values)
            for i, (train_index, test_index) in enumerate(kf):
                print('第{}次训练...'.format(i))
                train_feat1 = X.iloc[train_index]
                train_feat2 = X.iloc[test_index]
                xgb_train1 = xgb.DMatrix(train_feat1[columns].values, label=train_feat1['血糖'].values)
                xgb_train2 = xgb.DMatrix(train_feat2[columns].values, label=train_feat2['血糖'].values)
                model = xgb.train(param,
                                  xgb_train1,
                                  num_boost_round=3000,
                                  evals=[(xgb_train2,'valid')],
                                  verbose_eval=50,
                                  feval=mse,
                                  early_stopping_rounds=30)

                # scores.to_csv('score_%d.csv' % i,index=False)
                train_preds[test_index] += model.predict(xgb_train2)

                test_preds[:, i] = model.predict(xgb_test)

            print('线下得分：    {}'.format(mean_squared_error(X['血糖'], train_preds) * 0.5))

            submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
            submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                              index=False)

    def lgbmodel(self,istrain=False,iters=600):

        def mse(pred, df):
            label = df.get_label().values.copy()
            score = mean_squared_error(label, pred) * 0.5
            return ('mse', score, False)

        def define_loss(preds, dtrain):

            pass
            # return grad, hess

        X = self.train
        X['血糖']=self.label
        cols = list(X.columns)
        cols.remove('血糖')
        cate_feature=['性别']
        for col in cate_feature:
            X[col]=X[col].astype('category')
            self.test[col]=self.test[col].astype('category')

        params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'sub_feature': 0.2,
            'num_leaves': 2**9,
            'max_depth':30,
            'min_data': 100,
            'min_hessian': 1,
            'verbose': -1,
        }
        params['learning_rate']=0.01
        params['sub_feature']=0.5


        print(X.columns)
        if istrain==True:
            lgb_train = lgb.Dataset(X[cols], X['血糖'])
            model=lgb.train(params,
                            lgb_train,
                            num_boost_round=iters,
                            valid_sets=[lgb_train],
                            verbose_eval=50,
                            feval=mse,
                            early_stopping_rounds=30)
            feat_imp = pd.Series(model.feature_importance(), index=cols).sort_values(ascending=False)

            print(feat_imp)


            predict = model.predict(self.test)
            subm = pd.DataFrame(predict)
            subm.to_csv('submission%d.csv' % iters, index=False, header=False)

        else:


            train_preds = np.zeros(X.shape[0])
            test_preds = np.zeros((self.test.shape[0], 5))
            kf = KFold(len(X), n_folds=5, shuffle=True, random_state=520)

            average_score=pd.DataFrame()
            average_score['feature']=cols
            average_score['scores1']=0
            average_score['scores2']=0
            average_score['scores3']=0
            for i, (train_index, test_index) in enumerate(kf):
                print('第{}次训练...'.format(i))
                train_feat1 = X.iloc[train_index]
                train_feat2 = X.iloc[test_index]
                lgb_train1 = lgb.Dataset(train_feat1[cols], train_feat1['血糖'])
                lgb_train2 = lgb.Dataset(train_feat2[cols], train_feat2['血糖'])
                model = lgb.train(params,
                                lgb_train1,
                                num_boost_round=3000,
                                valid_sets=[lgb_train2],
                                verbose_eval=50,
                                feval=mse,
                                early_stopping_rounds=100)
                # feat_imp=pd.DataFrame()
                # feat_imp['feature']=cols
                scores = pd.DataFrame()
                scores['feature'] = cols
                scores['scores1'] = model.feature_importance()
                scores['scores2'] = model.feature_importance(importance_type='gain')
                scores['scores3'] = model.feature_importance(importance_type='gain') / model.feature_importance()

                average_score[['scores1','scores2','scores3']]+=scores[['scores1','scores2','scores3']]

                scores=scores.sort_values(by=['scores2', 'scores1', 'scores3'], ascending=False)
                # scores.to_csv('score_%d.csv' % i,index=False)
                train_preds[test_index] += model.predict(train_feat2[cols])

                test_preds[:, i] = model.predict(self.test)

            average_score[['scores1','scores2','scores3']]/=5


            # average_score.sort_values(by=['scores3', 'scores2', 'scores1'],
            #                           ascending=False).to_csv('score_ave_3.csv',index=False)
            # average_score.sort_values(by=['scores2', 'scores1', 'scores3'],
            #                           ascending=False).to_csv('score_ave_gain.csv', index=False)
            # average_score.sort_values(by=['scores1', 'scores3', 'scores2'],
            #                           ascending=False).to_csv('score_ave_split.csv', index=False)
            print(average_score.sort_values(by=['scores2', 'scores1', 'scores3'],ascending=False))
            print('线下得分：    {}'.format(mean_squared_error(X['血糖'], train_preds) * 0.5))

            submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
            submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                              index=False)

    def dnn(self):
        pass

def test():
    x=pd.read_csv('sub20180109_213103.csv',header=None)
    print(x.describe())
    y=pd.read_csv('sub20180109_211625.csv',header=None)
    print(y.describe())

#
a=model()
# a.lgbmodel(istrain=False,iters=850)
a.xgbmodel()



























