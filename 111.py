import xgboost as xgb
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import Counter

df = pd.read_csv('merged.csv',index_col=0)
print(df.head())
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)
print(label.head())
print(data.head())
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'num_class':3}
num_round = 10
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
xgb.plot_importance(bst)
#plt.show()
feature_importance = bst.get_score(importance_type='gain')
print(feature_importance)
k = Counter(feature_importance)
high = k.most_common(100)
print(high)