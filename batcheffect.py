 
  
'''
Using random forest to calculate accuracy
Compare the results using Kfold and cross_validation
'''
import statistics

from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


import patsy
import numpy as np
import sys

def limma(pheno, exprs, covariate_formula, design_formula='1', rcond=1e-8):
    design_matrix = patsy.dmatrix(design_formula, pheno)
    
    design_matrix = design_matrix[:,1:]
    rowsum = design_matrix.sum(axis=1) -1
    design_matrix=(design_matrix.T+rowsum).T
    
    covariate_matrix = patsy.dmatrix(covariate_formula, pheno)
    design_batch = np.hstack((covariate_matrix,design_matrix))
    coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)
    beta = coefficients[-design_matrix.shape[1]:]
    return exprs - design_matrix.dot(beta).T
    
pheno=pd.read_table('C:/Users/rucha/Desktop/GSE76312_Giustacchini-Thongjuea_et.al_Cell_Annotation.txt', index_col=0)
exprs=pd.read_table('C:/Users/rucha/Desktop/Giustacchini_Thongjuea_et.al_Nat.Med.RPKM.txt', index_col=0)
regressed=limma(pheno, exprs, "BCR_ABL_status")
regressed.to_csv('py-batch.txt',sep='\t')
df = pd.read_csv('py-batch.txt',index_col=0)
df = df[df.sign != 2]
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

#select features from csv file to remove redunant features
#can be commented if you want to use all features
'''feature_list = pd.read_csv('selected_features.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]
'''
scores = []
kf = KFold(n_splits=10)

#begin calculate accuracy for each fold
for train_index, test_index in kf.split(data):
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], label.iloc[train_index], label.iloc[test_index]
    #normalization here.
    sc_X = StandardScaler()

    clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)


print(statistics.mean(scores))

#begin calculate accuracy using cross_validation
clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)
svc_score = cross_validate(clf, data, label, cv=10, scoring='accuracy')
svc_results = pd.DataFrame.from_dict(svc_score)
svc_results.to_csv("rf_selected.csv")

