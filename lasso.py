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

scaler = StandardScaler()
scaler.fit(data.fillna(0))

selected = SelectFromModel(LogisticRegression(penalty='l1'))
selected.fit(scaler.transform(data.fillna(0)), label)

selected.get_support()

selected_feat = data.columns[(selected.get_support())]
print('total features: {}'.format((data.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(selected.estimator_.coef_ == 0)))
selected_feat = pd.DataFrame(selected_feat)

selected_feat.to_csv("selected_features.csv")


