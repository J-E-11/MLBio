'''
Author: Jinwan Huang
Calculate ROC AUC, precision,recall for xgboost clustering on 3 classes
And also on different feature numbers selected by xgboost
'''
from statistics import mean
import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def multiclass_roc_auc(truth, pred, average=None):
    lb = LabelBinarizer()
    lb.fit([0,1,2])
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    try:
        tmp = roc_auc_score(truth, pred, average=average)
    except:
        tmp = [np.nan, np.nan, np.nan]
    return tmp

def iter_(name):
    f = open('result.txt','a')
    df = pd.read_csv('merged.csv',index_col=0)
    print(df.head())
    label = df['sign']
    data = df.drop(['Cell', 'sign'], axis=1)

    feature_list = pd.read_csv(name+'.csv')
    feature_list = feature_list.drop(['score'],axis=1)
    feature_list = feature_list.values.tolist()
    feature_list = [k[0] for k in feature_list]
    data = data[feature_list]

    kf = KFold(n_splits=10)
    scores = []
    auc = []
    for train_index, test_index in kf.split(data):
        X_train, X_test, y_train, y_test = data.iloc[train_index], \
        data.iloc[test_index], label.iloc[train_index], label.iloc[test_index]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = {'max_depth': 5, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class':3}
        num_round = 50
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        bst = xgb.train(param, dtrain, num_round, evallist)
        ypred = bst.predict(dtest)
        
        accuracy = accuracy_score(y_test, ypred)
        scores.append(accuracy)
        roc_ = multiclass_roc_auc(y_test, ypred)
        auc.append(roc_)

    f.write(name+"\n\rscores:"+str(scores)+"\n\rauc:"+str(auc)+"\n\r"+str(mean(scores))+" " +str(np.nanmean(auc, axis=0)))
    print(scores)
    print(auc)
    print(mean(scores), np.nanmean(auc, axis=0))
    f.close()

f = open('result.txt','a')
df = pd.read_csv('merged.csv',index_col=0)
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

kf = KFold(n_splits=10)
scores = []
auc = []
for train_index, test_index in kf.split(data):
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], label.iloc[train_index], label.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'max_depth': 5, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class':3}
    num_round = 50
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    ypred = bst.predict(dtest)
    accuracy = accuracy_score(y_test, ypred)
    scores.append(accuracy)
    roc_ = multiclass_roc_auc(y_test, ypred)
    auc.append(roc_)
f.write("all features\n\rscores:"+str(scores)+"\n\rauc:"+str(auc)+"\n\r"+str(mean(scores))+" " +str(np.nanmean(auc, axis=0)))
print(scores)
print(auc)
print(mean(scores), np.nanmean(auc, axis=0))
f.close()
name_list = ['gene_score_15000', 'gene_score_12000', 'gene_score_10000', 'gene_score_8000', 'gene_score_6000', \
    'gene_score_4000', 'gene_score_3000', 'gene_score_2000', 'gene_score_1500', 'gene_score_1000', 'gene_score_800',\
         'gene_score_600', 'gene_score_500', 'gene_score_400', 'gene_score_300', 'gene_score_200', 'gene_score_100',\
             'gene_score_50']

for name in name_list:
    iter_(name)
