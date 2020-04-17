'''
Author: Jinwan Huang
'''
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


df = pd.read_csv('merged.csv',index_col=0)
label = df['sign']
data = df.drop(['Cell', 'sign'], axis=1)


feature_list = pd.read_csv('gene_score.csv')
feature_list = feature_list.drop(['score'],axis=1)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]


pca = PCA(n_components=20)
principalComponents = pca.fit_transform(data)
kmeans = KMeans(n_clusters=3).fit_predict(principalComponents)
y_pred = kmeans
y_true = label.values
print("The result of KMeans using PCA is \n")
print(label.describe())
print(confusion_matrix(y_true, y_pred))

print("-----------------------------------")
kmeans = KMeans(n_clusters=3).fit_predict(data)
y_pred = kmeans
y_true = label.values
print("The result of KMeans without using PCA is \n")
print(label.describe())
print(confusion_matrix(y_true, y_pred))
