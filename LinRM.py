#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:27:12 2022

@author: Rachel Perry
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import scale

#%%
# PROBLEM 1 [6 POINTS]
# For this problem, you are going to use the Airline Costs dataset.  
headers = ["Airline",
           "Flight Length", 
           "MPH", 
           "Daily Flight Time",
           "Population",
           "Operating Costs",
           "Revenue",
           "Load Factor",
           "Available Capacity",
           "Total Assets",
           "Investments",
           "Adjusted Assets"]
ACd = pd.read_csv("/Users/rachel/Desktop/IMT 574 Datasets/Assignment 2—Problem 1_ airline_costs.csv", names = headers)
print(ACd.head())
# Download Airline Costs dataset.The dataset has the following attributes:
    # Airline
    # Length of flight (miles) -> predictor variable
    # Speed of plane (miles per hour)
    # Daily flight time per plane (hours) -> predictor variable
    # Population served (1000s) -> predicted variable
    # Total operating cost (cents per revenue ton-mile)
    # Revenue tons per aircraft mile
    # Ton-mile load factor (proportion)
    # Available capacity (tons per mile)
    # Total assets ($100,000s)
    # Investments and special funds ($100,000s)
    # Adjusted assets ($100,000s)

# Part A [3 Points]

# Use a linear regression model to predict the number of customers each 
# airline serves from its length of the flight and daily flight time per plane.
# RESOURCE: https://realpython.com/linear-regression-in-python/#multiple-linear-regression-with-scikit-learn

x = np.array([ACd["Flight Length"], ACd["Daily Flight Time"]]).reshape(-1,2)
y = np.array([ACd["Population"]]).reshape(-1,1)
# print(x)
# print(y)

# Report your model (linear equation). -- METHOD 1
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
    # 0.072 - measure of fit of the model
print('intercept:', model.intercept_)
    # 13946.38
print('slope:', model.coef_)
    # -53.20660422(Flight Length),  72.03428958(Daily Flight Time)

# METHOD 2 (FROM CLASS DEMO) - verified results
# x = sm.add_constant(x)
# lr_model = sm.OLS(y,x).fit()
# print(lr_model.summary())

# Model Regression Line:
    # y = -53.20660422(x) + 72.03428958(x) + 13946.38
print('Model Regression Line: y = -53.20660422(x) + 72.03428958(x) + 13946.38')

# What is your predicted value for the number of customers served 
# for a flight that is 200 miles in length and 
# has a daily flight time per plane of 7.2 hours? 
print('predicted value - number of customers served (in 1000s):', -53.20660422*(200) + 72.03428958*(7.2) + 13946.38)
    # predicted value: 3823.7060409759997
 
# Part B [3 Points]

# Next, build another regression model to predict 
# the total assets of an airline from the customer served by the airline.
x2 = np.array([ACd["Population"]]).reshape(-1,1)
y2 = np.array([ACd["Total Assets"]]).reshape(-1,1)

# Once again, report your model. - METHOD 1
model2 = LinearRegression().fit(x2, y2)
r_sq2 = model2.score(x2, y2)
print('coefficient of determination:', r_sq2)
    # 0.08186 - measure of fit of the model
print('intercept:', model2.intercept_)
    # -98.50798581
print('slope:', model2.coef_)
    # 0.02165468(Total Assets)

# METHOD 2 (FROM CLASS DEMO) - verified results
# x2 = sm.add_constant(x2)
# lr_model2 = sm.OLS(y2,x2).fit()
# print(lr_model2.summary())

# Model Regression Line:
    # y = 0.02165468(x) -98.50798581
print('Model Regression Line: y = 0.02165468(x) -98.50798581')


# What is your prediction for total assets for an airline, 
# given they serve 20,300,000 customers? 
print('predicted value - total assets (in $100,000s):', 0.02165468*(20300) - 98.50798581)
    # predicted value: 341.08201819

#%%
# PROBLEM 2 [4 POINTS]
# In this exercise, you are going to use the kangaroo’s nasal dimension data. 
# Download kangaroo’s nasal dimension data.
KNDd = pd.read_excel("/Users/rachel/Desktop/IMT 574 Datasets/Assignment 2—Problem 2_ kangaroo’s nasal dimension data.xls", engine='xlrd')
# plt.scatter(KNDd['X'], KNDd['Y'])

# Use the gradient descent algorithm to predict the optimal intercept 
# and gradient for this problem. Report your gradient values.

X = KNDd[['X']]
X = sm.add_constant(X)
y = KNDd[['Y']]

KNDd_lr_model = sm.OLS(y,X).fit()
print(KNDd_lr_model.summary())

# Linear Model
    # Y = 0.2876X + 46.4508
# Plot data points 
    # plt.scatter(KNDd['X'], KNDd['Y']) 
# Plot regression line
    # Y_pred = (0.2876)*(KNDd['X']) + (46.4508)
    # plt.plot([min(KNDd['X']), max(KNDd['X'])], [min(Y_pred), max(Y_pred)], color='red')  # regression line
    # plt.show()

# Use the gradient descent algorithm to predict the optimal intercept 
# and gradient for this problem. Report your gradient values.

def grad_descent_class_demo(X, y, alpha, epsilon):
    iteration = [0] # counts number of steps
    i = 0 # first interval
    theta = np.ones(shape=(len(KNDd.columns),1)) # set general array
    cost = [np.transpose(X @ theta - y) @ (X @ theta - y)] # evaluate the error at each point
    delta = 1
    while (delta>epsilon):
        theta = theta - alpha*((np.transpose(X)) @ (X @ theta - y)) # move to the next point
        cost_val = (np.transpose(X @ theta - y)) @ (X @ theta - y) # calculate the error
        cost.append(cost_val) # add the error value to the arrary
        delta = abs(cost[i+1]-cost[i]) # update delta per the error difference
        if ((cost[i+1]-cost[i]) > 0 ):
            print("cost is increasing, reduce alpha")
            break
        iteration.append(i) # step counter
        i += 1 # go to the next step
    print("Completed in %d iterations." %(i))
    return(theta)

# X = pd.concat((pd.DataFrame([1,len(KNDd)]), KNDd[['X']]), axis=1, join='outer').to_numpy()
    # the above was from the example but I didn't understand the case for adjusting the X array
X = X.to_numpy()
y = y.to_numpy()
     

theta = grad_descent_class_demo(X = preprocessing.scale(X), y=y, alpha = 0.02, epsilon = 10**-10)
print("Theta: ", theta)
#  Theta = [1, 26.34297605]





