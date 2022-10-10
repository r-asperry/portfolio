#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Rachel Perry
Class: IMT 574

Assignment 4

In this exercise, you will work with the “Blues Guitarists Hand Posture and 
Thumbing Style by Region and Birth Period” data 
This dataset has 93 entries of various blues guitarists born 
between 1874 and 1940. Apart from the name of the guitarists, that dataset 
contains the following four features:
    Regions: 
        1 means East, 
        2 means Delta, 
        3 means Texas
    Years: 
        0  = <1906, 
        1 = >1906
    Hand postures: 
        1= Extended, 
        2= Stacked, 
        3=Lutiform
    Thumb styles: Between 1 and 3, 
        1=Alternating
        2=Utility
        3=Dead

Step 1: Using decision tree on this dataset, how accurately you can tell 
their birth year from their hand postures and thumb styles. 
How does it affect the evaluation when you include the region 
while training the model?

Step 2: Now do the same using random forest (in both of the above cases) 
and report the difference. Make sure to use appropriate training-testing 
parameters for your evaluation.

You should also run the algorithms multiple times, measure various accuracies, 
and report the average (and perhaps the range).
"""
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# STEP 1 - DECISION TREE (code from class example)
# Predict Birth Year (Y), using Hand Posture and Thumb Style (X)

# Import Data
blues_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/Assignment 4-blues_hand.csv')

# define variables
X = blues_data[['handPost','thumbSty']]

# predicting post1906 as opposed to brthYr
# brthYr is a numerical value and post1906 is organizse as a binary which is
# better suited for classification problems
y = blues_data[['post1906']]

s1_report1 = []
s1_report1 = pd.DataFrame(s1_report1, columns = ['i', 'accuracy'])

# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(100):
   # create train and test data split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   # Make the decision tree and fit it to the training data
   dtree = DecisionTreeClassifier()
   # fit the decision tree model
   dtree.fit(X_train, y_train)
   # make predictions
   dt_predictions = dtree.predict(X_test)
   s1_report1 = s1_report1.append({'i':x,
                            'accuracy':accuracy_score(y_test, dt_predictions)}, 
                           ignore_index = True)

# report the rules of the given decision tree
# text_representation = tree.export_text(dtree)
# print(text_representation)

# Predict Birth Year (Y), using Hand Posture, Thumb Style and Region (X)
X2 = blues_data[['handPost','thumbSty','region']]

s1_report2 = []
s1_report2 = pd.DataFrame(s1_report2, columns = ['i', 'accuracy'])
# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(100):
   # create train and test data split
   X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, 
                                                           test_size=0.2)
   # Make the decision tree and fit it to the training data
   dtree2 = DecisionTreeClassifier()
   # fit the decision tree model
   dtree2.fit(X2_train, y2_train)
   # make predictions
   dt_predictions2 = dtree2.predict(X2_test)
   s1_report2 = s1_report2.append({'i':x,
                            'accuracy':accuracy_score(y2_test, 
                                                      dt_predictions2)}, 
                           ignore_index = True)

# report the rules of the given decision tree
# text_representation = tree.export_text(dtree2)
# print(text_representation)

#%%
# STEP 2 - RANDOM FOREST (code from class example)
# Predict Birth Year (Y), using Hand Posture and Thumb Style (X)

s2_report1 = []
s2_report1 = pd.DataFrame(s2_report1, columns = ['i', 'accuracy'])

# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(100):
    # create train and test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Random Forest from elass example
    rfc = RandomForestClassifier(n_estimators=100)
    # reshaping y_train to match n_samples
    rfc.fit(X_train, y_train.values.ravel())
    # make predictions
    rf_predictions = rfc.predict(X_test)
    s2_report1 = s2_report1.append({'i':x,
                            'accuracy':accuracy_score(y_test, 
                                                      rf_predictions)}, 
                           ignore_index = True)

# Predict Birth Year (Y), using Hand Posture, Thumb Style and Region (X)
s2_report2 = []
s2_report2 = pd.DataFrame(s2_report2, columns = ['i', 'accuracy'])
# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(100):
    # create train and test data split
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, 
                                                            test_size=0.2)
    rfc2 = RandomForestClassifier(n_estimators=100)
    # reshaping y_train to match n_samples
    rfc2.fit(X2_train, y2_train.values.ravel())
    # make predictions
    rf2_predictions = rfc2.predict(X2_test)
    s2_report2 = s2_report2.append({'i':x,
                            'accuracy':accuracy_score(y2_test,
                                                      rf2_predictions)}, 
                           ignore_index = True)

#%%
# REPORTING RESULTS
# DT1
dt1 = s1_report1['accuracy']
print("Decision Tree - Hand and Thumb Predictors")
print("Average Accuracy:", format(statistics.mean(dt1), ".3f"))
print("Accuracy Range:", format(min(dt1), ".3f"), "-", format(max(dt1),".3f"))
# RF1
rf1 = s2_report1['accuracy']
print("Random Forest - Hand and Thumb Predictors")
print("Average Accuracy:", format(statistics.mean(rf1), ".3f"))
print("Accuracy Range:", format(min(rf1), ".3f"),"-",format(max(rf1),".3f"))
# VAR
print("DT/FT Accuracy Variance:", 
      format(statistics.mean(rf1) - statistics.mean(dt1), ".3f"))
# DT2 
dt2 = s1_report2['accuracy']
print("Decision Tree - Hand, Thumb and Region Predictors")
print("Average Accuracy:", format(statistics.mean(dt2),".3f"))
print("Accuracy Range:", 
      format(min(dt2),".3f"), "-",
      format(max(dt2),".3f"))
# RF2
rf2 = s2_report2['accuracy']
print("Random Forest - Hand, Thumb and Region Predictors")
print("Average Accuracy:",format(statistics.mean(rf2),".3f"))
print("Accuracy Range:", format(min(rf2),".3f"),"-",format(max(rf2),".3f"))
# VAR
print("DT/FT Accuracy Variance:", 
      format(statistics.mean(rf2) - statistics.mean(dt2), ".3f"))

"""
Decision Tree - Hand and Thumb Predictors
Average Accuracy: 0.557
Accuracy Range: 0.158 - 0.789

Random Forest - Hand and Thumb Predictors
Average Accuracy: 0.585
Accuracy Range: 0.316 - 0.789

DT/FT Accuracy Variance: 0.028

Decision Tree - Hand, Thumb and Region Predictors
Average Accuracy: 0.542
Accuracy Range: 0.316 - 0.789

Random Forest - Hand, Thumb and Region Predictors
Average Accuracy: 0.556
Accuracy Range: 0.263 - 0.789

DT/FT Accuracy Variance: 0.014

"""
