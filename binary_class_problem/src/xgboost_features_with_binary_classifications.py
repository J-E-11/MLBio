'''@author Kaur, Sukhleen'''

from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate)

#for original data
'''df = pd.read_csv('merged.csv',index_col=0)
df2 = df[df.sign != 2]

label = df2['sign']
data = df2.drop(['Cell', 'sign'], axis=1)'''

#for data with batch effect removed
df = pd.read_csv('merged_rm_be.csv')
df = df.drop(df.columns[[0]], axis=1)
df2 = df[df.sign!=2]
cleaned_data = df2.dropna()

label = cleaned_data['sign']
data = cleaned_data.drop(['sign'], axis=1)

data_matrix = xgb.DMatrix(data, label)

params = {'max_depth': 5, 'eta': 0.1, 'objective':'binary:logistic'}

trained = xgb.train(params, data_matrix)
#xgb.plot_importance(trained)
feature_importance = trained.get_score(importance_type='weight')
k = Counter(feature_importance)
feature_importance_table = pd.DataFrame.from_dict(k, orient='index').reset_index()
feature_importance_table.columns = ['Features', 'Weights']

#feature importance
fts = feature_importance_table['Features']
fts = pd.DataFrame.from_dict(fts)
#fts.to_csv('features_from_xgboost_binary.csv') #features on original data
fts.to_csv('features_from_xgboost_binary_batch.csv')
weights = feature_importance_table['Weights']

list_selected_features = fts['Features'].to_list()
data_f = data[np.intersect1d(data.columns, list_selected_features)]


skfold = StratifiedKFold(n_splits=10, random_state=None)

##LASSO Classification
print('Doing LASSO Classification')
clf_lasso = LogisticRegression(penalty='l1', solver='liblinear')
score_lasso = cross_validate(clf_lasso, data_f, label, cv=skfold, scoring=['accuracy', 'precision', 'recall', 'f1'])
lasso_results = pd.DataFrame.from_dict(score_lasso)
print('Lasso Acc: %.3f' % lasso_results['test_accuracy'].mean(),
      'Lasso Precision: %.3f' % lasso_results['test_precision'].mean(),
      'Lasso Recall: %.3f' % lasso_results['test_recall'].mean(),
      'Lasso F1: %.3f' % lasso_results['test_f1'].mean())
y_pred_lasso = cross_val_predict(clf_lasso, data_f, label, cv=skfold)

y_pred_lasso_prob = cross_val_predict(clf_lasso, data_f, label, cv=skfold, method='predict_proba')
y_pred_lasso_df = pd.DataFrame.from_dict(y_pred_lasso_prob)
y_pred_lasso_df.columns = ['Diagnosis', 'Remission']
y_pred_lasso_POS = y_pred_lasso_df['Remission']

##Conf Matrix LASSO
cmtx_svm_Lasso = pd.DataFrame(
    confusion_matrix(label, y_pred_lasso, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_Lasso, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Lasso")

##SVM 
'''does not seem to stop running (probably cant use the features provided)
print('Doing SVM Classification') 
model_svm = svm.SVC(kernel='linear', probability=True)
svc_score = cross_validate(model_svm, data_f, label, cv=skfold, scoring=['accuracy', 'precision', 'recall', 'f1'])
svc_results = pd.DataFrame.from_dict(svc_score)
print('SVM Acc: %.3f' % svc_results['test_accuracy'].mean(),
      'SVM Precision: %.3f' % svc_results['test_precision'].mean(),
      'SVM Recall: %.3f' % svc_results['test_recall'].mean(),
      'SVM F1: %.3f' % svc_results['test_f1'].mean())
y_pred_svc = cross_val_predict(model_svm, data_f, label, cv=skfold)

y_pred_svc_prob = cross_val_predict(model_svm, data_f, label, cv=skfold, method='predict_proba')
y_pred_svc_df = pd.DataFrame.from_dict(y_pred_svc_prob)
y_pred_svc_df.columns = ['Diagnosis', 'Remission']
y_pred_svc_POS = y_pred_svc_df['Remission']

##Conf Matrix SVM
cmtx_svm_SVC = pd.DataFrame(
    confusion_matrix(label, y_pred_svc, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_SVC, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM")'''

##XGBoost
print('Doing XGBoost Classification')
xgb_clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, objective='binary:logistic')
score_xgb = cross_validate(xgb_clf, data_f, label, cv=skfold, scoring=['accuracy', 'precision', 'recall', 'f1'])
xgb_results = pd.DataFrame.from_dict(score_xgb)
print('XGB Acc: %.3f' % xgb_results['test_accuracy'].mean(),
      'XGB Precision: %.3f' % xgb_results['test_precision'].mean(),
      'XGB Recall: %.3f' % xgb_results['test_recall'].mean(),
      'XGB F1: %.3f' % xgb_results['test_f1'].mean())
y_pred_xgb = cross_val_predict(xgb_clf, data_f, label, cv=skfold)

y_pred_xgb_prob = cross_val_predict(xgb_clf, data_f, label, cv=skfold, method='predict_proba')
y_pred_xgb_df = pd.DataFrame.from_dict(y_pred_xgb_prob)
y_pred_xgb_df.columns = ['Diagnosis', 'Remission']
y_pred_xgb_POS = y_pred_xgb_df['Remission']

#Conf Matrix XGB
cmtx_svm_XGB = pd.DataFrame(
    confusion_matrix(label, y_pred_xgb, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_XGB, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for XGBoost")

##RandomForest
print('Doing Random Forest Classification')
rf_clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)
score_rf = cross_validate(rf_clf, data_f, label, cv=skfold, scoring=['accuracy', 'precision', 'recall', 'f1'])
rf_results = pd.DataFrame.from_dict(score_rf)
print('RF Acc: %.3f' % rf_results['test_accuracy'].mean(),
      'RF Precision: %.3f' % rf_results['test_precision'].mean(),
      'RF Recall: %.3f' % rf_results['test_recall'].mean(),
      'RF F1: %.3f' % rf_results['test_f1'].mean())
y_pred_rf = cross_val_predict(rf_clf, data_f, label, cv=skfold)

y_pred_rf_prob = cross_val_predict(rf_clf, data_f, label, cv=skfold, method='predict_proba')
y_pred_rf_df = pd.DataFrame.from_dict(y_pred_rf_prob)
y_pred_rf_df.columns = ['Diagnosis', 'Remission']
y_pred_rf_POS = y_pred_rf_df['Remission']

#Conf Matrix RF
cmtx_svm_RF = pd.DataFrame(
    confusion_matrix(label, y_pred_rf, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_RF, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for RF")



##ROC Curve
auc_svm = roc_auc_score(label, y_pred_svc_POS)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(label, y_pred_svc_POS)

auc_lasso = roc_auc_score(label, y_pred_lasso_POS)
fpr_lasso, tpr_lasso, thresholds_lasso = roc_curve(label, y_pred_lasso_POS)

auc_xgb = roc_auc_score(label, y_pred_xgb_POS)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(label, y_pred_xgb_POS)

auc_rf = roc_auc_score(label, y_pred_rf_POS)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(label, y_pred_rf_POS)


#plt.plot(fpr_svm, tpr_svm, label='SVM = %0.2f' % auc_svm, color="#", linewidth=2) #SVM discarded since it did not run
plt.plot(fpr_lasso, tpr_lasso, label='Lasso = %0.2f' % auc_lasso, color="#ffcc00", linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label='XGB = %0.2f' % auc_xgb, color="#5dbcd2", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label='RF = %0.2f' % auc_rf, color="#cb42f5", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color="#878787")  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver Operating Characteristic', fontsize=18)
plt.legend(loc="lower right", title="Area Under Curve")
