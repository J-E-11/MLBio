'''
Author: Jinwan Huang
'''
import xgboost as xgb
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report


df = pd.read_csv('merged.csv',index_col=0)
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 5, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class':3}
num_round = 50
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
accuracy = accuracy_score(y_test, ypred)

feature_importance = bst.get_score(importance_type='weight')
feature_importance_table = pd.DataFrame.from_dict(k, orient='index').reset_index()
feature_importance_table.columns = ['Features', 'Weights']

fts = feature_importance_table['Features']
fts = pd.DataFrame.from_dict(fts)
fts.to_csv('features_from_xgboost.csv')
weights = feature_importance_table['Weights']