import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LassoCV

df = pd.read_csv('merged.csv',index_col=0)
#print(df.head())
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)
#print(label.head())
#print(data.head())

standardized_data = preprocessing.scale(data)

lasso_scores = LassoCV(cv=10).fit(standardized_data, label)
