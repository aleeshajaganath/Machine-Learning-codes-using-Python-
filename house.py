
    #2. Writa a program to predict house price using regression and find mean square error


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import svm
from imblearn.over_sampling import ADASYN
from sklearn.cross_validation import train_test_split
import pandas as pd
ad =ADASYN()
df1= pd.read_csv('Assignment 2 dataset.csv',sep=',')
df=df1[0:1000]
train,test= train_test_split(df,test_size=0.20)
x_train=train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
'condition','grade','sqft_above','sqft_basement','long','sqft_living15','sqft_lot']]
x_test=test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
'condition','grade','sqft_above','sqft_basement','long','sqft_living15','sqft_lot']]
y_train=train['price']
y_test=test['price']
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
rm=reg.score(x_test,y_test)
print(" rmse for regression")
rmse=mean_squared_error(y_test,y_pred)
print(rmse)
clf = svm.SVR()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("predicted y value")
print(y_pred)
rmse=mean_squared_error(y_test,y_pred)
print(" rmse for svm")
print(rmse)
