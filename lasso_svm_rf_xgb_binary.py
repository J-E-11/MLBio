import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from sklearn import svm
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt
import xgboost as xgb

df = pd.read_csv('merged.csv',index_col=0)
df2 = df[df.sign != 2]

label = df2['sign']
data = df2.drop(['Cell', 'sign'], axis=1)

##LASSO feature selection
std_data = preprocessing.scale(data)
model = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))
model.fit(std_data, label)
model.get_support()


selected_feat = data.columns[(model.get_support())]
selected_feat = pd.DataFrame.from_dict(selected_feat)
print('total features: {}'.format((data.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(model.estimator_.coef_ == 0)))


########

selected_feat.columns = ['genes']
selected_feat.to_csv('selected_features_binary.csv')
list_selected_features = selected_feat['genes'].to_list()
data_f = data[np.intersect1d(data.columns, list_selected_features)]

##LASSO Classification
clf_lasso = LogisticRegression(penalty='l1', solver='liblinear')
score_lasso = cross_validate(clf_lasso, data_f, label, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
lasso_results = pd.DataFrame.from_dict(score_lasso)
y_pred_lasso = cross_val_predict(clf_lasso, data_f, label, cv=10)

##Conf Matrix
cmtx_svm_Lasso = pd.DataFrame(
    confusion_matrix(label, y_pred_lasso, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_Lasso, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Logistic Regression with Lasso Parameter")

##SVM
model_svm = svm.SVC(kernel='linear')
svc_score = cross_validate(model_svm, data_f, label, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
svc_results = pd.DataFrame.from_dict(svc_score)
y_pred_svc = cross_val_predict(model_svm, data_f, label, cv=10)

##Conf Matrix
cmtx_svm_SVC = pd.DataFrame(
    confusion_matrix(label, y_pred_svc, normalize='true', labels=[0, 1]),
    index=['Diagnosis', 'Remission'],
    columns=['Diagnosis', 'Remission']
)

ax = sns.heatmap(cmtx_svm_SVC, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM")

##XGBoost
xgb_clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, objective='binary:logistic')
score_xgb = cross_validate(xgb_clf, data_f, label, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
xgb_results = pd.DataFrame.from_dict(score_xgb)
y_pred_xgb = cross_val_predict(xgb_clf, data_f, label, cv=10)

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
rf_clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=0)
score_rf = cross_validate(rf_clf, data_f, label, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
rf_results = pd.DataFrame.from_dict(score_rf)
y_pred_rf = cross_val_predict(rf_clf, data_f, label, cv=10)

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
auc_svm = roc_auc_score(label, y_pred_svc)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(label, y_pred_svc)

auc_lasso = roc_auc_score(label, y_pred_lasso)
fpr_lasso, tpr_lasso, thresholds_lasso = roc_curve(label, y_pred_lasso)

auc_xgb = roc_auc_score(label, y_pred_xgb)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(label, y_pred_xgb)

auc_rf = roc_auc_score(label, y_pred_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(label, y_pred_rf)


plt.plot(fpr_svm, tpr_svm, label='SVM = %0.2f' % auc_svm, color="#ffcc00", linewidth=2)
plt.plot(fpr_lasso, tpr_lasso, label='Lasso = %0.2f' % auc_lasso, color="#5dbcd2", linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label='XGB = %0.2f' % auc_xgb, color="#cb42f5", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label='RF = %0.2f' % auc_rf, color="#f54242", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color="#878787")  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver Operating Characteristic', fontsize=18)
plt.legend(loc="lower right", title="Area Under Curve")