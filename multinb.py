# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:24:56 2020

@author: rucha
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Read dataset from csv
dataset = pd.read_csv("merged.csv")
dataset= dataset[dataset.sign != 2]
label = dataset['sign']
data = dataset.drop(['Cell', 'sign'], axis=1)
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
print(dataset.head())

# Features
#features = ['Day','Month','Year','Humidity','Max Temperature','Min Temperature',
   #         'Rainfall','Sea Level Pressure','Sunshine','Wind Speed']
#target = 'Cloud'




#select features from csv file to remove redunant features
#can be commented if you want to use all features
feature_list = pd.read_csv('selected_features.csv', index_col=0)
feature_list = feature_list.values.tolist()
feature_list = [k[0] for k in feature_list]
data = data[feature_list]











x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                    train_size=0.7, test_size=0.3, shuffle=False)

# Print samples after running train_test_split
print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

print("\n")

# Multinomial Naive Bayes Model setup after parameter tuning
model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
model.fit(x_train, y_train)

# Print results to evaluate model
print("Showing Performance Metrics for Naive Bayes Multinomial\n")
print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
predicted = model.predict(x_test)
print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("\n")

print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)

print("\n")

print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))

print("\n")

print("Confusion Matrix for Naive Bayes Multinomial\n")
labels = [0, 1, 2]
cm = confusion_matrix(y_test, predicted, labels=labels)
print(cm)

print("\n")

print('Precision, Recall and f-1 Scores for Naive Bayes Multinomial\n')
print(classification_report(y_test, predicted))