#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rachel Perry

In this exercise, you will use the Portuguese sea battles data that contains 
outcomes of naval battles between Portuguese and Dutch/British ships 
between 1583 and 1663.

The dataset has the following features:
Battle	Name of the battle place
Year	Year of the battle
Portuguese ships	Number of Portuguese ships
Dutch ships	Number of Dutch ships
English ships	Number of ships from English side
The ratio of Portuguese to Dutch/British ships	
Spanish Involvement	1=Yes, 0=No
Portuguese outcome	-1=Defeat, 0=Draw, 1=Victory
 

INSTRUCTIONS
1) Use an SVM-based model to predict the Portuguese outcome of the battle from 
the number of ships involved on all sides and Spanish involvement.
    y = outcome

2) Try solving the same problem using two other classifiers that you know.
    1 - kNN
    2 - Random Forest

3) Report and compare their results with those from SVM.

Step 1: An SVM-based model is used to predict the Portuguese outcome of the 
battle from the number of ships involved in all sides and Spanish involvement.
Step 2: Two other classifiers are used to solve the same problem.
Step 3: Results are reported and compared with those from SVM.

"""
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from matplotlib import pyplot


# add headers
headers =["Battle",
          "Year",
          "Portuguese ships",
          "Dutch ships",
          "English ships",
          "P:D/B Ratio",
          "Spanish Involvement", 
          "Portuguese Outcome"]
# read in csv
data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/armada.csv', 
                   names = headers)
#  pre-processing
data = data.dropna()
#  no rows removed
X = data.drop(['Portuguese Outcome','Battle', 'Year'], axis = 1)
y = data['Portuguese Outcome']
# TRAIN/TEST 80/20 SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#%%
#  MODEL 1 - SVM
model1 = Pipeline([('scale', StandardScaler()),
                 ('scv', SVC(kernel='poly', degree=3))])
# FIT
model1.fit(X_train, y_train)
# PREDICT
svc_predict = model1.predict(X_test)
# REPORT MODEL METRICS
print(accuracy_score(y_test,svc_predict))
print(confusion_matrix(y_test,svc_predict))
# print(classification_report(y_test,svc_predict))

"""
LINEAR KERNEL RESULTS
0.333
[[0 2 0]
 [0 2 0]
 [0 2 0]]
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         2
           0       0.33      1.00      0.50         2
           1       0.00      0.00      0.00         2

    accuracy                           0.33         6
   macro avg       0.11      0.33      0.17         6
weighted avg       0.11      0.33      0.17         6

POLY DEGREE 2 RESULTS
0.833
[[1 0 0]
 [0 4 1]
 [0 0 0]]
              precision    recall  f1-score   support

          -1       1.00      1.00      1.00         1
           0       1.00      0.80      0.89         5
           1       0.00      0.00      0.00         0

    accuracy                           0.83         6
   macro avg       0.67      0.60      0.63         6
weighted avg       1.00      0.83      0.91         6
"""
#%%
# MODEL 2 - kNN
# FIND BEST CLUSTER VALUE -- 2
# report1 = []
# report1 = pd.DataFrame(report1, columns = ['k', 'accuracy'])
# for x in range(2,10):
#     knn = KNeighborsClassifier(n_neighbors=x)
#     knn.fit(X_train, y_train)
#     knn_predict = knn.predict(X_test)
#     # store accuracy per each k value in output dataframe
#     report1 = report1.append({'k':x,
#                             'accuracy':accuracy_score(y_test,knn_predict)}, 
#                              ignore_index = True)
# pyplot.plot(report1['k'],report1['accuracy'])
# pyplot.title('KNN Accuracy Values')
# pyplot.xlabel('k value')
# pyplot.ylabel('Accuracy')
# pyplot.show()

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
knn_predict = knn.predict(X_test)

print(accuracy_score(y_test,knn_predict))
print(confusion_matrix(y_test,knn_predict))
print(classification_report(y_test,knn_predict))
"""
kNN 2 NEIGHBORS RESULTS
0.666
[[1 0]
 [2 3]]
              precision    recall  f1-score   support

          -1       0.33      1.00      0.50         1
           0       1.00      0.60      0.75         5

    accuracy                           0.67         6
   macro avg       0.67      0.80      0.62         6
weighted avg       0.89      0.67      0.71         6
"""
#%%
# MODEL 3 - RANDOM FOREST
report2 = []
report2 = pd.DataFrame(report2, columns = ['i', 'accuracy'])

# loop 20 iterations to calculate an average accuracy for each data shuffle
for x in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train.values.ravel())
    rf_predict = rfc.predict(X_test)
    report2 = report2.append({'i':x,
                              'accuracy':accuracy_score(y_test,
                                                        rf_predict)},
                             ignore_index = True)
rf = report2['accuracy']
print("Average Accuracy:", format(statistics.mean(rf), ".3f"))
print("Accuracy Range:", format(min(rf), ".3f"),"-",format(max(rf),".3f"))

"""
RANDOM FOREST RESULTS
Average Accuracy: 0.387
Accuracy Range: 0.167 - 0.833
"""
