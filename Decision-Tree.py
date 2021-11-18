# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:41:12 2019

@author: DELL
"""

# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('CDR.xlsx')
X = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc= (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,1]+cm[1,0])
sensitivity=cm[0,0]/(cm[0,0]+cm[1,0])
specificity=cm[1,1]/(cm[1,1]+ cm[0,1])
print( acc)
print(specificity)
print(sensitivity)


