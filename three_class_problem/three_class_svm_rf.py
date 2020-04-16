'''
Author: Jinwan Huang
Three class classification using SVM and Random Forest
return scores and confusion matrix
containing both before batch effect removal and after 
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



df = pd.read_csv('merged.csv',index_col=0)
#print(df.head())
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

feature_list = pd.read_csv('features_from_xgboost.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]

########

model = svm.SVC(kernel='linear')

score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
#print("score ", score)
results = pd.DataFrame.from_dict(score)
#print(results)
print('SVM Acc: %.3f' % results['test_accuracy'].mean(),
      'SVM Precision: %.3f' % results['test_precision_macro'].mean(),
      'SVM Recall: %.3f' % results['test_recall_macro'].mean(),
      'SVM F1: %.3f' % results['test_f1_macro'].mean())

y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
print(cm)
cm = cm/sum(cm)



a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM Without Batch Effect")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('SVM_cm.png')
plt.clf()
######

model = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)

score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
results = pd.DataFrame.from_dict(score)
print('RF Acc: %.3f' % results['test_accuracy'].mean(),
      'RF Precision: %.3f' % results['test_precision_macro'].mean(),
      'RF Recall: %.3f' % results['test_recall_macro'].mean(),
      'RF F1: %.3f' % results['test_f1_macro'].mean())

y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
print(cm)
cm = cm/sum(cm)

a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Random Forest Without Batch Effect")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('rf_cm.png')
plt.clf()

#for data with batch effect removed
df = pd.read_csv('merged_rm_be.csv')
df = df.drop(df.columns[[0]], axis=1)

cleaned_data = df.dropna()

label = cleaned_data['sign']
data = cleaned_data.drop(['sign'], axis=1)

feature_list = pd.read_csv('features_from_xgboost_rm_be.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]

########

model = svm.SVC(kernel='linear')

score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
#print("score ", score)
results = pd.DataFrame.from_dict(score)
#print(results)
print('SVM+Batch effect removal Acc: %.3f' % results['test_accuracy'].mean(),
      'SVM+Batch effect removal Precision: %.3f' % results['test_precision_macro'].mean(),
      'SVM+Batch effect removal Recall: %.3f' % results['test_recall_macro'].mean(),
      'SVM+Batch effect removal F1: %.3f' % results['test_f1_macro'].mean())

y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
cm = cm/sum(cm)



a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM After Batch Effect")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('SVM_cm_rm_be.png')
plt.clf()
######

model = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)

score = cross_validate(model, data, label, cv=10, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
results = pd.DataFrame.from_dict(score)
print('RF+Batch effect removal Acc: %.3f' % results['test_accuracy'].mean(),
      'RF+Batch effect removal Precision: %.3f' % results['test_precision_macro'].mean(),
      'RF+Batch effect removal Recall: %.3f' % results['test_recall_macro'].mean(),
      'RF+Batch effect removal F1: %.3f' % results['test_f1_macro'].mean())

y_pred = cross_val_predict(model, data, label, cv=10)
cm = confusion_matrix(label, y_pred)
print(cm)
cm = cm/sum(cm)

a = pd.DataFrame(data=cm, index=['negative','positive','normal'],columns=['negative','positive','normal'])
ax = sns.heatmap(a, cmap="YlGnBu", annot=True, square=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Random Forest After Batch Effect")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('rf_cm_rm_be.png')















