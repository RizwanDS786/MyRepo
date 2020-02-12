## Created by Rizwanullah Shaik

#####                                                             Code for Credit Card Fraud Detection model....                                                 ###

## Created a Classifier model which will train on past data which is already classified based on features and can classify the upcoming transactions as fraud or fair
## We have used Support Vector Classifier with Poly Kernel for our model
## Also used grid search model to tune the hyper parameters of Support vector machine( gamma and loss) 


# Importing the required libraries.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.use("Agg")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
import random

dataset = pd.read_csv('creditcard.csv')
data_11 = dataset[dataset.Class == 0].iloc[1:5000, :]
data_22 = dataset[dataset.Class == 1].iloc[1:5000, :]

data = pd.concat([data_11,data_22])

data1 = dataset[dataset.Class == 0].iloc[1:500, :]
data2 = dataset[dataset.Class == 1].iloc[1:100, :]

dataGrid = pd.concat([data1,data2])

print(data1.shape)
print(data2.shape)
print(data.shape)

random.seed(2)
pred_columns = data[:]
pred_columns.drop(['Class'], axis = 1, inplace = True)

# Prediction_var contains the columns to be used as features

prediction_var = pred_columns.columns
print(list(prediction_var))

train_G, test_G = train_test_split(dataGrid, test_size = 0.3)

train_XG = train_G[prediction_var]
train_YG = train_G['Class']

train, test = train_test_split(data, test_size = 0.3)

train_X = train[prediction_var]
train_Y = train['Class']
test_X = test[prediction_var]
test_Y = test['Class']
print(train_XG.shape)
print(train_YG.shape)
print(train_X.shape)

import numpy as np
from sklearn.model_selection import GridSearchCV
clf = svm.SVC(kernel='poly')

# Range of gamma values to be used in grid search

gamma_val = [0.001,0.01,0.1,1]

# Range of C values to be used in grid search

c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'gamma': gamma_val}

# Finding the best hyper parameters, It runs the model for every combination of parameters. Where ever it gets high accuracy , will be identified as best parameters.
clf_cv = GridSearchCV(clf,param_grid,cv=5)
clf_cv.fit(train_XG,train_YG)

# Best parameters for the model with given data 
print(clf_cv.best_params_)

#Best accuracy obtained with hyper parameter tuning
print((clf_cv.best_score_))

# Creating a model with kernel as 'Poly'
clf = svm.SVC(kernel='poly', C = 0.001, gamma = 1e-05)
clf.fit(train_X,train_Y)
Test = pd.read_csv('TestData.csv')
print(Test.head())
print(" \n Label for the given data : {} ".format(clf.predict(Test)[0]))