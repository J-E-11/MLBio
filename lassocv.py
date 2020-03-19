# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:08:35 2020

@author: rucha
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import KFold
from sklearn import linear_model

dataset = pd.read_csv("C:/Users/rucha/Desktop/merged.csv")
X = dataset['sign']
y= dataset.drop(['Cell', 'sign'], axis=1)

X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.25, random_state=0)

alphas = np.logspace(-4, -1, 10)
scores = np.empty_like(alphas)
for i,a in enumerate(alphas):
    lasso = linear_model.Lasso()
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)
    print(a, lasso.coef_)
    
lassocv = linear_model.LassoCV()
lassocv.fit(X, y)
lassocv_score = lassocv.score(X, y)
lassocv_alpha = lassocv.alpha_
print('CV', lassocv.coef_)

plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color='b', ls='--')
plt.axvline(lassocv_alpha, color='b', ls='--')
plt.xlabel(r'$\alpha$')
plt.ylabel('Score')
plt.xscale('log')
sns.despine(offset=15)