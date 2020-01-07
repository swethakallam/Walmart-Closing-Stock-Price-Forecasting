#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 03:26:59 2019

@author: beingpratiksahoo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from pylab import rcParams
import sklearn
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale # for scaling the data
import sklearn.metrics as sm # for evaluating the model
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report

#import dataset
nyse = pd.read_csv("/Users/beingpratiksahoo/Desktop/Spring19/Multivariate/Project/fundamentals.csv")

#drop NA rows, removing the 1st 3 parameters and the target features
nyse.dropna(inplace = True)
nyse1 = nyse.iloc[:,3:75]
eps = nyse.iloc[:,-2]
eps = eps.astype('int')

#checking for value range of different features for feature scaling
nyse1.apply(lambda x: pd.Series([x.min(), x.max()])).T.values.tolist()

#scaling all the features to a similar ranges 
scaler = MinMaxScaler()
nyse_scaled =  pd.DataFrame(scaler.fit_transform(nyse1),columns=nyse1.columns)

#checking for value range of different features after feature scaling
nyse_scaled.apply(lambda x: pd.Series([x.min(), x.max()])).T.values.tolist()

#Feature Selection
#1 chisq univariate
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(nyse_scaled,eps)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(nyse_scaled.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 10 best features

#2 extratree classifier
model = ExtraTreesClassifier()
model.fit(nyse_scaled,eps)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=nyse_scaled.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#3 randomforest
rf_exp = SelectFromModel(RandomForestClassifier(n_estimators= 100, random_state=100))
rf_exp.fit(nyse_scaled,eps)
rf_exp.get_support()
selected_feat= nyse_scaled.columns[(rf_exp.get_support())]
print(nyse_scaled.columns[rf_exp.get_support()])

nyse_scaled1 = nyse_scaled[['Other Operating Items','Pre-Tax Margin','Profit Margin','Operating Margin','Long-Term Investments','Non-Recurring Items','Common Stocks','Depreciation','Accounts Payable','Cost of Revenue','Total Current Liabilities','Total Revenue','Other Liabilities','Net Receivables','Research and Development','Total Current Assets','Inventory','Deferred Liability Charges','Total Liabilities & Equity','Total Assets']]
eps1 = nyse.iloc[:,75:76]
nyse_scaled1_train = nyse_scaled1.iloc[0:1199,:]
eps1_train = eps1.iloc[0:1199,:]
nyse_scaled1_test = nyse_scaled1.iloc[1200:1299,:]
eps1_test = eps1.iloc[1200:1299,:]


nyse_scaled1.to_csv(r'/Users/beingpratiksahoo/Desktop/Spring19/Multivariate/Project/nyse_scaled1.csv', index = None, header=True)
eps1.to_csv(r'/Users/beingpratiksahoo/Desktop/Spring19/Multivariate/Project/eps1.csv', index = None, header=True)

AAL_train = nyse_scaled1.iloc[8:11,:]
AALeps_train = eps1.iloc[8:11,:]
AAL_test = nyse_scaled1.iloc[11:12,:]
AALeps_test = eps1.iloc[11:12,:]

#clustering
nyse_cluster = pd.read_csv("/Users/beingpratiksahoo/Desktop/Spring19/Multivariate/Project/Cluster_Data.csv")
kmeans=KMeans(4)

kmeans.fit(nyse_cluster[['Total Assets','Long-Term Debt']])
clusters3=nyse_cluster[['Total Assets','Long-Term Debt']].copy()
clusters3['clusters_predict_assetsvsdebt']=kmeans.fit_predict(nyse_cluster[['Total Assets','Long-Term Debt']])


nyse_cluster_companies = nyse_cluster.loc[ [8,9,10,11,80,81,82,124,125,126,423,424,425,1207,1208,1208,1209] , : ]

"""
name = nyse ['Ticker Symbol']
name_df = pd.DataFrame(name)

nyse.drop(nyse.columns[[0,1]], axis = 1, inplace = True)

ss = MinMaxScaler()
nyse_scale =  pd.DataFrame(ss.fit_transform(nyse),columns=nyse.columns)

EPS = nyse_scale.filter(['Earnings Per Share'], axis = 1)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(EPS)
EPS = pd.DataFrame(training_scores_encoded)

nyse_scale.drop(['Earnings Per Share'], axis = 1)

rf_exp = SelectFromModel(RandomForestClassifier(n_estimators= 100, random_state=100))
rf_exp.fit(nyse_scale,EPS )

rf_exp.get_support()

selected_feat= nyse_scale.columns[(rf_exp.get_support())]
len(selected_feat)

print(nyse_scale.columns[rf_exp.get_support()])

train_data = pd.DataFrame(nyse_scale.iloc[ :1200])
test_data =  pd.DataFrame(nyse_scale.iloc[ 1200:])
train_label = pd.DataFrame(name_df.iloc[ :1200])
test_label = pd.DataFrame(name_df.iloc[1200:])
train_EPS = pd.DataFrame(EPS.iloc[ :1200])
test_EPS = pd.DataFrame(EPS.iloc[1200:])

features_train = train_data.filter(['Accounts Receivable', 'Addl income/expense items', 'After Tax ROE',
       'Capital Expenditures', 'Cash Ratio', 'Cash and Cash Equivalents',
       'Changes in Inventories', 'Current Ratio', 'Depreciation',
       'Effect of Exchange Rate', 'Fixed Assets', 'Income Tax', 'Investments',
       'Liabilities', 'Net Borrowings', 'Net Cash Flow',
       'Net Cash Flows-Financing', 'Net Income Adjustments', 'Other Assets',
       'Other Current Assets', 'Other Equity', 'Other Financing Activities',
       'Other Investing Activities', 'Other Operating Activities',
       'Pre-Tax ROE', 'Quick Ratio', 'Retained Earnings',
       'Sale and Purchase of Stock', 'Sales, General and Admin.',
       'Total Current Assets', 'Total Equity', 'Total Liabilities',
       'Earnings Per Share', 'Estimated Shares Outstanding'], axis=1)
features_test = test_data.filter(['Accounts Receivable', 'Addl income/expense items', 'After Tax ROE',
       'Capital Expenditures', 'Cash Ratio', 'Cash and Cash Equivalents',
       'Changes in Inventories', 'Current Ratio', 'Depreciation',
       'Effect of Exchange Rate', 'Fixed Assets', 'Income Tax', 'Investments',
       'Liabilities', 'Net Borrowings', 'Net Cash Flow',
       'Net Cash Flows-Financing', 'Net Income Adjustments', 'Other Assets',
       'Other Current Assets', 'Other Equity', 'Other Financing Activities',
       'Other Investing Activities', 'Other Operating Activities',
       'Pre-Tax ROE', 'Quick Ratio', 'Retained Earnings',
       'Sale and Purchase of Stock', 'Sales, General and Admin.',
       'Total Current Assets', 'Total Equity', 'Total Liabilities',
       'Earnings Per Share', 'Estimated Shares Outstanding'], axis=1)

rf_exp = RandomForestRegressor(n_estimators= 100, random_state=100)
rf_exp.fit(features_train,train_EPS)

r_sq = rf_exp.score(features_train,train_EPS)
#print(r_sq)
predictions = rf_exp.predict(features_test)
#print("predictions are - ", predictions, sep='\n')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_EPS, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_EPS, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_EPS, predictions)))

LR = LinearRegression().fit(features_train,train_EPS)
r_sq_LR = LR.score(features_train,train_EPS)
#print(r_sq_LR)
predict_LR = LR.predict(features_test)
#print("predictions are - ", predict_LR, sep='\n')"""