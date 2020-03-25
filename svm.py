import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


from sklearn import svm

df = pd.read_csv('merged.csv',index_col=0)

label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)

df2 = pd.read_csv("selected_features.csv")
df2.columns = ['index', 'genes']
df2 = df2.drop(['index'], axis=1)

list = df2['genes'].to_list()

data_f = df[np.intersect1d(df.columns, list)]

########

model = svm.SVC(kernel='linear')
svc_score = cross_validate(model, data_f, label, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
svc_results = pd.DataFrame.from_dict(svc_score)
y_pred = cross_val_predict(model, data_f, label, cv=10)
#auc_svm = roc_auc_score(label, y_pred, average='weighted', multi_class='ovo')
#svc_auc = pd.DataFrame.from_dict(auc_svm)

#svc_results.to_csv('svc_results.csv')
#svc_auc.to_csv('svc_auc.csv')


cmtx_svm = pd.DataFrame(
    confusion_matrix(label, y_pred, normalize='true', labels=[0, 1, 2]),
    index=['Diagnosis', 'Remission', 'HSC'],
    columns=['Diagnosis', 'Remission', 'HSC']
)

ax = sns.heatmap(cmtx_svm, annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM")

######

def multiclass_roc_auc(truth, pred, average=None):
    lb = LabelBinarizer()
    lb.fit([0,1,2])

    truth = lb.transform(truth)
    pred = lb.transform(pred)
    #print(truth)
    #print(pred)

    try:
        tmp = roc_auc_score(truth, pred, average=average)
    except:
        tmp = [np.nan, np.nan, np.nan]
    return tmp















