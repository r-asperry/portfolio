#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:18:29 2022

@author: rachel
"""
import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

"""
PROBLEM 1 [5 POINTS]
An automated answer-rating site marks each post in a community forum 
website as “good” or “bad” based on the quality of the post. 
The CSV file contains the various types of quality as measured by the tool. 
The following lists the type of qualities that the dataset contains:

Types of qualities - Description
i. num_words - 
    number of words in the post
ii. num_characters - 
    number of characters in the post
iii. num_misspelled - 
    number of misspelled words
iv. bin_end_qmark - 
    if the post ends with a question mark
v. num_interrogative - 
    number of interrogative words in the post
vi. bin_start_small - 
    if the answer starts with a lowercase letter ("1" means yes, otherwise no)
vii. num_sentences - 
    number of sentences per post
viii. num_punctuations - 
    number of punctuation symbols in the post
ix. label
     the label of the post ("G" for good and "B" for bad)
"""

# INSTRUCTIONS
# 1 - Create a logistics regression model to predict the class label from the 
# first eight attributes of the question set.
qual_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/Assignment 3-Problem 1—quality.csv')
print(qual_data.head())

# independent vars = everything except label
X = qual_data.drop('label', axis = 1)
# transform the G / B label binary to 1 and 0
qual_data['label'] = np.where(qual_data['label'] == 'G', 1, 0)
# label becomes dependent
y = qual_data['label']
#  separate the data into train and test sets - 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# create logistic regression model
log_model = LogisticRegression()
# fit the model to the training data
log_model.fit(X_train, y_train)
# making predictions on the Test set
predictions = log_model.predict(X_test)

print('LR Classification report on test data: ')
print(classification_report(y_test, predictions))
print('LR Accuracy report on test data: ')
print(accuracy_score(y_test, predictions))
# 1.0, all predictions are accurate
print('LR Confusion matrix on test data: ')
print(confusion_matrix(y_test, predictions))
# no false attributions

#%%
# 2 - Try doing the same using two different subsets 
# (your choice) of those eight attributes.
# For the two subsets that you use, provide some justification 
# (why you chose those features in a given subset).

"""
X2 subset gives metrics for aspects of the writing style that could contribute 
to quality, in terms of mispellings, the type of words used, and how the 
writing sample began and ended
"""
X2 = qual_data[['num_misspelled',
                'num_interrogative',
                'bin_start_small',
                'bin_end_qmark']]

"""
X3 subset gives metrics for aspects of the contents of the sample that could
contribute to quality, in terms of the number of words and how that translates 
to the number of characters and sentences as well as the number of punctuations
"""
X3 = qual_data[['num_words',
                'num_characters',
                'num_sentences', 
                'num_punctuations']]

# 3- Report the accuracies of each of these three models.
# As discussed, it is useful to report not just a single accuracy number 
# for a given model, but either an average accuracy over many runs or a 
# distribution of accuracies over those runs.

# reporting values over multiple iterations for subset 2
# inialize report datafram to store results for average accuracy
report2 = []
report2 = pd.DataFrame(report2, columns = ['i', 'accuracy'])

# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(20):
    X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.30)
    # create logistic regression model
    log_model2 = LogisticRegression()
    # fit the model to the training data
    log_model2.fit(X2_train, y_train)
    # making predictions on the Test set
    predictions2 = log_model2.predict(X2_test)
    # add results to the report dataframe
    report2 = report2.append({'i':x,
                            'accuracy':accuracy_score(y_test,predictions2)}, 
                           ignore_index = True)
# print the mean of the accuracy scores
print(statistics.mean(report2['accuracy']))
    # X2 average accuracy: 0.38

# reporting values over multiple iterations for subset 3
# inialize report datafram to store results for average accuracy
report3 = []
report3 = pd.DataFrame(report3, columns = ['i', 'accuracy'])

# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(20):
    X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.30)
    # create logistic regression model
    log_model3 = LogisticRegression()
    # fit the model to the training data
    log_model3.fit(X3_train, y_train)
    # making predictions on the Test set
    predictions3 = log_model3.predict(X3_test)
    # add results to the report dataframe
    report3 = report3.append({'i':x,
                            'accuracy':accuracy_score(y_test,predictions3)}, 
                           ignore_index = True)
# print the mean of the accuracy scores
print(statistics.mean(report3['accuracy']))
    # X3 average accuracy: 0.6

#%%
"""
PROBLEM 2 [5 POINTS]
Using a “wine” dataset, containing information about several wines, 
their characteristics, and their quality, do some experiments 
(trial-and-error) to figure out a good subset of features to use for 
learning wine quality. Report these features.
"""

# INSTRUCTIONS
# Download the wine dataset. It contains information about several 
# wines—their characteristics (features) and if it's considered 
# high quality or not (1 or 0).
wine_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/Assignment 3-Problem 2—wine dataset.csv')
print(wine_data.head())

# First, do some experiments (trial-and-error) to figure out a 
# good subset of features to use for learning wine quality (last column). 
# Report these features.

X5 = wine_data.drop(['high_quality','color'], axis=1)
# dropping the color category because is_red feature captures this binary
y5 = wine_data['high_quality']

# Feature exploration adapted from "Logistic Regression Feature Importance"
# https://machinelearningmastery.com/calculate-feature-importance-with-python/
model5 = LogisticRegression()
model5.fit(X5, y5)
importance = model5.coef_[0]
# report the pseudo-metric for importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# Features for analysis based on strength on scores
    # Quality +9.4
    # Alcohol +5.6
    # Volatile_acidity -4.5
    #  ~ Residual_sugar +2.9
    #  ~ Chlorides -2.5
    #  ~ Sulphates +2.6

# normalization technique from class example
normalized = X5[['volatile_acidity',
                 'residual_sugar',
                 'chlorides',
                 'sulphates',
                 'alcohol',
                 'quality']].apply(lambda x: (x -min(x))/(max(x)-min(x)))

#%%
# Then, use 70% data for training to build a kNN classifier with 
# different values of k ranging from 2–10.

# KNN Technique from class example
# generate test and training subsets
X5_train, X5_test, y5_train, y5_test = train_test_split(normalized, y5, 
                                                      test_size=0.30)

# initalize output dataframe for results
output = []
output = pd.DataFrame(output, columns = ['k', 'accuracy'])

# iterate through the desired K values to show varying accuracy
for x in range(2,10):
    # Create KNN Classifier Model
    knn = KNeighborsClassifier(n_neighbors=x)
    # Fit the training data
    knn.fit(X5_train, y5_train)
    #  make predictions
    predict = knn.predict(X5_test)
    #  print values to verify for loop is producing results
    print(confusion_matrix(y5_test,predict))
    print(accuracy_score(y5_test,predict))
    # store accuracy per each k value in output dataframe
    output = output.append({'k':x,
                            'accuracy':accuracy_score(y5_test,predict)}, 
                           ignore_index = True)
    
# Plot your accuracies with each of these. In other words, your final result 
# will be a line chart with k on the x-axis and accuracy on the y-axis. 
from matplotlib import pyplot

pyplot.plot(output['k'],output['accuracy'])
pyplot.title('KNN Accuracy Values')
pyplot.xlabel('k value')
pyplot.ylabel('Accuracy')
pyplot.show()
