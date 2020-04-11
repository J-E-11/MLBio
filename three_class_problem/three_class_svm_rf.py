'''
Author: Jinwan Huang
Three class classification using SVM and Random Forest
return scores and confusion matrix
'''
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

df = pd.read_csv('merged_rm_be.csv',index_col=0)
label = df['sign']
data = df.drop(['label','sign'], axis=1)

feature_list = pd.read_csv('features_from_xgboost.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]

########

model = svm.SVC(kernel='linear')
score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
results = pd.DataFrame.from_dict(score)
print('SVM Acc: %.3f' % results['test_accuracy'].mean(),
      'SVM Precision: %.3f' % results['test_precision'].mean(),
      'SVM Recall: %.3f' % results['test_recall'].mean(),
      'SVM F1: %.3f' % results['test_f1'].mean())
y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
print(cm)
cm = cm/sum(cm)

a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('SVM_cm.png')
######

model = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)
score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
results = pd.DataFrame.from_dict(score)
print('RF Acc: %.3f' % results['test_accuracy'].mean(),
      'RF Precision: %.3f' % results['test_precision'].mean(),
      'RF Recall: %.3f' % results['test_recall'].mean(),
      'RF F1: %.3f' % results['test_f1'].mean())
y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
print(cm)
cm = cm/sum(cm)

a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Random Forest")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('rf_cm.png')















