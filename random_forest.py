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


df = pd.read_csv('merged.csv',index_col=0)
df = df[df.sign != 2]
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

#select features from csv file to remove redunant features
#can be commented if you want to use all features
feature_list = pd.read_csv('selected_features.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]

scores = []
kf = KFold(n_splits=10)

#begin calculate accuracy for each fold
for train_index, test_index in kf.split(data):
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], label.iloc[train_index], label.iloc[test_index]
    #normalization here.
    sc_X = StandardScaler()
    #X_train = sc_X.fit_transform(X_train)
    #X_test = sc_X.transform(X_test)
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