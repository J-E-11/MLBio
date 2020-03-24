from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

df = pd.read_csv('merged.csv',index_col=0)
print(df.head())
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)
print(label.head())
print(data.head())
feature_list = pd.read_csv('gene_score.csv')
#feature_list.reset_index(drop=True, inplace=True)
feature_list = feature_list.drop(['score'],axis=1)
print(feature_list.head())
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
print(feature_list)
data = data[feature_list]
#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
print(data.head())
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(data)
kmeans = KMeans(n_clusters=3).fit_predict(principalComponents)
y_pred = kmeans
y_true = label.values
print(label.describe())
print(confusion_matrix(y_true, y_pred))
