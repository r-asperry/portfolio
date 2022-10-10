#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Rachel Perry

PROBLEM STATEMENT
For this assignment, we are going to use the Steel Plates Faults Dataset as 
available from here.

INSTRUCTIONS
For this exercise use a neural network and see how well you could predict the 
type of faults in steel plates from numeric attributes only.

X - all numeric attributes
y - type of steel plate fault (1 of 7 dependent variables)

Note: To save time and energy use the hidden layer numbers and number of nodes 
in hidden layers that your computer can handle.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#%%
headers = ['X_Minimum',
           'X_Maximum',
           'Y_Minimum',
           'Y_Maximum',
           'Pixels_Areas',
           'X_Perimeter',
           'Y_Perimeter',
           'Sum_of_Luminosity',
           'Minimum_of_Luminosity',
           'Maximum_of_Luminosity',
           'Length_of_Conveyer',
           'TypeOfSteel_A300',
           'TypeOfSteel_A400',
           'Steel_Plate_Thickness',
           'Edges_Index',
           'Empty_Index',
           'Square_Index',
           'Outside_X_Index',
           'Edges_X_Index',
           'Edges_Y_Index',
           'Outside_Global_Index',
           'LogOfAreas',
           'Log_X_Index',
           'Log_Y_Index',
           'Orientation_Index',
           'Luminosity_Index',
           'SigmoidOfAreas',
           'Pastry',
           'Z_Scratch',
           'K_Scratch',
           'Stains',
           'Dirtiness',
           'Bumps',
           'Other_Faults']

sp_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/Assignment 9 dataset-Faults.csv', 
                      delimiter=('\t'), names = headers)
X = pd.DataFrame(sp_data.iloc[:,0:27])
# Among the independent variables, only the steel types (12th and 13th) are 
# categorical variables, the rest are numeric.
# Use a neural network and see how well you could predict the type of faults 
# in steel plates from numeric attributes only.
    # Remove the columns for TypeOfSteel_A300, and TypeOfSteel_A400
X = X.drop(['TypeOfSteel_A300','TypeOfSteel_A400'], axis =1)

# Collapse all the fault types into one column
sp_data['Fault'] = 0
sp_data.loc[sp_data.iloc[:,27] == 1, ['Fault']] = 1
sp_data.loc[sp_data.iloc[:,28] == 1, ['Fault']] = 2
sp_data.loc[sp_data.iloc[:,29] == 1, ['Fault']] = 3
sp_data.loc[sp_data.iloc[:,30] == 1, ['Fault']] = 4
sp_data.loc[sp_data.iloc[:,31] == 1, ['Fault']] = 5
sp_data.loc[sp_data.iloc[:,32] == 1, ['Fault']] = 6
sp_data.loc[sp_data.iloc[:,33] == 1, ['Fault']] = 7

y = sp_data['Fault']
y1 = sp_data['Pastry']
y2 = sp_data['Z_Scratch']
y3 = sp_data['K_Scratch']
y4 = sp_data['Stains']
y5 = sp_data['Dirtiness']
y6 = sp_data['Bumps']
y7 = sp_data['Other_Faults']

#%%
# if the analysis was meant to include the categorical variables, would use
# LabelEncoder() to fit and transform the categorical variable columns to 
# preprocess the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

nn = Sequential()
nn.add(Dense(27, activation='relu'))
nn.add(Dense(12, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)
y_pred = (y_pred>0.5)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

"""
RESULTS
Y1 Accuracy: 0.9254498714652957
Confusion matrix:
[360   0]
[ 29   0]

Y2 Accuracy: 0.9023136246786633
Confusion matrix:
[349   1]
[ 37   2]

Y3 Accuracy: 0.922879177377892
Confusion matrix:
[290  14]
[ 16  69]

Y4 Accuracy: 0.9640102827763496
Confusion matrix:
[375   0]
[ 14   0]

Y5 Accuracy: 0.9665809768637532
Confusion matrix:
[376   0]
[ 13   0]

Y6 Accuracy: 0.7455012853470437
Confusion matrix:
[287  12]
[ 87   3]

Y7 Accuracy: 0.5167095115681234
Confusion matrix:
[123 111]
[ 77  78]

Y Accuracy (i.e predicting 7 classes of faults at once)
Edititing the activation functio nand density values, obtained an
Accuracy range from: 0.069-0.089.

Overall the best accuracty scores were on the y4 and y5 data sets which 
represented predictions for Stains and Dirtiness Faults. Therefore we can 
conclude that from this data set of 25 attirbutes (omitting the two categorical 
types of steel variables) is best used to predict the Stains and Dirtiness 
types of Faults
"""







