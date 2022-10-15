#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rachel
"""
import pandas as pd
import numpy as np
import statistics
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

"""
PROBLEM STATEMENT
With the accidents involving Boeing's 737 Max, there have been speculations 
and concerns about airline safety. Then, there was a helicopter crash involving 
Kobe Bryant. Academic studies have found that high-profile crashes can shift 
passenger demand away from the airlines involved in the disasters.

Should travelers avoid airlines that have had crashes in the past? That is the 
question we will try to address in this exercise. The dataset for this has been 
sourced from Aviation Safety Network and available at this link: 
https://www.kaggle.com/fivethirtyeight/fivethirtyeight-airline-safety-dataset. 

DATASET
The dataset has the following list of attributes:

Attribute Name	Attribute Description
airline	Airline (asterisk indicates that regional subsidiaries are included)
avail_seat_km_per_week	Available seat kilometers flew every week
incidents_85_99	Total number of incidents, 1985–1999
fatal_accidents_85_99	Total number of fatal accidents, 1985–1999
fatalities_85_99	Total number of fatalities, 1985–1999
incidents_00_14	Total number of incidents, 2000–2014
fatal_accidents_00_14	Total number of fatal accidents, 2000–2014
fatalities_00_14	Total number of fatalities, 2000–2014
 
INSTRUCTIONS
Step 1: Use this dataset and two different clustering approaches 
(agglomerative and divisive) to group the airlines with similar safety records.

Step 2: Do these two approaches lead to the same/similar results? 
Provide appropriate visualizations, clustering summaries, and your 
interpretations.
"""
air_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/airline-safety.csv')

X = air_data[["incidents_85_99", "fatal_accidents_85_99", "fatalities_85_99",
              "incidents_00_14"	,"fatal_accidents_00_14","fatalities_00_14"]]
scaler = StandardScaler()
scaler.fit_transform(X)

# set variables for clusters
cl = 5

# k MEANS
kmeans = KMeans(n_clusters=cl)
y_means = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
# print(centroids)
print('K Means cludtering assignments', y_means)

# visualize the clusters
plt.figure(figsize=(10,10))
plt.title('KMeans / Divisive')
plt.scatter(X[['fatalities_85_99']],X[['fatalities_00_14']], 
            c=y_means, cmap='rainbow')
plt.scatter(centroids[:,0], centroids[:,2], c='black', s=100)
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative')
Dendrogram = sch.dendrogram((sch.linkage(X, method ='ward')))

# AGGLOMERATIVE
ac = AgglomerativeClustering(n_clusters=cl)
y_ac = ac.fit_predict(X)
print('Agglomerative Clustering Assignments', y_ac)

# EVALUATE THE TWO METHODS
air_data.insert(0,'k Means Class', y_means, True)
air_data.insert(0,'Agglomerative Class', y_ac, True)
air_data.insert(0,'Variance', y_means - y_ac, True)
print('accuracy score:', accuracy_score(y_means,y_ac))
print('average variance:', statistics.mean(air_data['Variance']))
# accuracy score: 0.03571428571428571
# average variance: -0.2857142857142857  

kmean_results = pd.DataFrame(air_data.groupby(['k Means Class']).mean())
kmean_results = kmean_results[["incidents_85_99", 
                               "fatal_accidents_85_99", 
                               "fatalities_85_99",
                               "incidents_00_14",
                               "fatal_accidents_00_14",
                               "fatalities_00_14"]]

ac_results = pd.DataFrame(air_data.groupby(['Agglomerative Class']).mean())
ac_results = ac_results[["incidents_85_99", 
                               "fatal_accidents_85_99", 
                               "fatalities_85_99",
                               "incidents_00_14",
                               "fatal_accidents_00_14",
                               "fatalities_00_14"]]
print(kmean_results.head())
print(ac_results.head())

# print(air_data[['k Means Class','Agglomerative Class','Variance']])
"""
Based on the an analysis of the average values for each of the categories
these two methods assigned similar classifications for the data:
    k Means 0 = Agglomerative 4
    k Means 1 = Agglomerative 1
    k Means 2 = Agglomerative 2
    k Means 3 = Agglomerative 0    
    k Means 4 = Agglomerative 3

So although the method and assignment of the data to the categories occurred
by different means, the acctual class differentiations are very similar for
these methods when analyzing this dataset.
"""



