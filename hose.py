# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:50:30 2018

@author: user
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import svm
#from sklearn.model_selection import cross_validate
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
df= pd.read_csv('Assignment 2 dataset.csv',sep=',')
#train1 = df.drop(['id', 'price'],axis=1)
train,test= train_test_split(df,test_size=0.20)
x_train=train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','long','sqft_living15','sqft_lot']]
x_test=test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','long','sqft_living15','sqft_lot']]
y_train=train['price']
y_test=test['price']
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
reg.fit(x_train,y_train)
rm=reg.score(x_test,y_test)
rmse=mean_squared_error(y_test,y_pred)
#sclf = svm.SVC()
print(rmse)
#print('root mean squared error')
#print(rmse)
#clf = svm.SVC()
#clf.fit(x_train,y_train)
#p=clf.predict(x_test)
