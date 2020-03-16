import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('merged.csv',index_col=0)
#print(df.head())
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)
#print(label.head())
#print(data.head())
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

selected = SelectFromModel(LogisticRegression(penalty='l1'))
selected.fit(scaler.transform(X_train.fillna(0)), y_train)

selected.get_support()

selected_feat = X_train.columns[(selected.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(selected.estimator_.coef_ == 0)))
selected_feat = pd.DataFrame(selected_feat)

selected_feat.to_csv("selected_features.csv")


