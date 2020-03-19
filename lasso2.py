import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('merged.csv',index_col=0)

label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

df2 = pd.read_csv("selected_features.csv")
df2.columns = ['index', 'genes']
df2 = df2.drop(['index'], axis=1)

list = df2['genes'].to_list()

data_f = df[np.intersect1d(df.columns, list)]

########

model = linear_model.Lasso()
lasso_cv = cross_val_score(model, data_f, label, cv=10, scoring='neg_mean_squared_error')
lasso_cv_scores = pd.DataFrame(lasso_cv)
y_pred_lasso = cross_val_predict(model, data_f, label, cv=10)

auc_lasso = roc_auc_score(label, y_pred_lasso)

######

model = LassoCV(cv=10).fit(data_f, label)















