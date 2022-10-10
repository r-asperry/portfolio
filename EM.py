#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Rachel Perry

PART 2 [8 POINTS]
The following data were obtained from Belsley, Kuh, and Welsch (1980). 
They in turn obtained the data from Sterling (1977).

The dataset contains 50 observations with five variables.
Number	Variables
I	Sr: numeric, aggregate personal savings
II	pop15: numeric, % of the population under 15
III	pop75: numeric, % of the population over 75
IV	dpi: numeric, real per-capita disposable income
V	ddpi: numeric, % growth rate of dpi

INSTRUCTIONS
Use EM for clustering “similar” countries.
Report how many groups you got and why you chose that number with the help of 
AIC and BIC. 
"""
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

lcs_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/Assignment 7 data-lifecyclesaving.csv')
lcs_data = lcs_data.rename(columns={"Contry": "Country"})

data = lcs_data.drop('Country', axis=1)
scaler = StandardScaler()
scaler.fit_transform(data)

report = []
report = pd.DataFrame(report, columns = ['cluster','epoch', 'AIC','BIC'])
# number of clusters 1-19
for n in range(10,11):
    # number of epochs 
    for i in range(1,100):
        model = GaussianMixture(n_components=4, init_params='random', max_iter=50)
        model.fit(data)
        yhat = model.predict(data)
        # print(i,"epoch:",n,"cluster")
        # print(yhat)
        # AIC
        # print(model.aic(data))
        # BIC
        # print(model.bic(data))
        report = report.append({'cluster':n,
                                'epoch':i,
                                'AIC':model.aic(data),
                                'BIC':model.bic(data)},ignore_index = True)

# print(report.groupby('cluster').min())
print(report.groupby('epoch').min())
aic = report['AIC']
print("AIC Range:", 
      format(min(aic),".3f"), "-",
      format(max(aic),".3f"))

bic = report['BIC']
print("BIC Range:",  
      format(min(bic),".3f"), "-",
      format(max(bic),".3f"))

"""
RESULTS:
        epoch          AIC          BIC
cluster                                 
2.0       40.0  1632.244207  1790.942116
3.0       40.0  1601.108217  1759.806126
4.0       40.0  1658.657014  1817.354923
5.0       40.0  1610.506221  1769.204131
6.0       40.0  1634.705226  1793.403135
7.0       40.0  1603.269944  1761.967854
8.0       40.0  1599.027334  1757.725244
9.0       40.0  1613.624893  1772.322803
10.0*     40.0  1550.210542  1708.90845
11.0      40.0  1643.261288  1801.959197
12.0      40.0  1593.305052  1752.002961
13.0      40.0  1643.473193  1802.171103
14.0      40.0  1581.147272  1739.845182
15.0      40.0  1620.090745  1778.788654
16.0      40.0  1636.140671  1794.838580
17.0      40.0  1651.537865  1810.235774
18.0      40.0  1616.388260  1775.086169
19.0      40.0  1600.630870  1759.328780
       cluster          AIC          BIC
epoch                                   
40.0       2.0  1636.140671  1794.838580
41.0       2.0  1599.534180  1758.232089
42.0       2.0  1593.305052  1752.002961
43.0       2.0  1621.421919  1780.119829
44.0       2.0  1600.630870  1759.328780
45.0       2.0  1601.108217  1759.806126
46.0       2.0  1581.147272  1739.845182
47.0*      2.0  1550.210542  1708.908451
48.0       2.0  1641.507937  1800.205847
49.0       2.0  1610.256233  1768.954142
AIC Range: 1550.211 - 1742.563
BIC Range: 1708.908 - 1901.261

CONCLUSION:
After exploring various value ranges for both the clusters and epochs, I 
decided to give a wide range for the cluster values whe nprinting the minimums
to show that all of the cluster options are operating within a small margin of 
one another. For the data reported above the lowest AIC and BIC scores occured 
with 10 clusters, indicating that this cluster value produced the model with
the best fit. BIC is higher than AIC because there is a "penalty" for
additional parameters (meaning it is biased towards more simple models).
I also was playing around with the number of epochs and none
of my value ranges dramatically changed the AIC or BIC scores so I chose a 
random value range on one of my previous runs with the lowest AIC and BIC 
for the reported values.

# of Clusters: 10
# of Epochs: 47
"""
