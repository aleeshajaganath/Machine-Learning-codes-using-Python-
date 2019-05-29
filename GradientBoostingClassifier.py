# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:41:24 2018

@author: user
"""
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
df1= pd.read_csv('breast-cancer-wisconsin.csv',dtype={"h":float,"y":float,"z":float,"a":float})#,"b":float,"c":float,"d":float,"e":float,"f":float,"g":float})
df=df1[1:650]
train,test= train_test_split(df,test_size=0.20)
yt=np.array(train[['h']]).ravel()
#print(y_train)
xt=train[['y','z','a']]#,'b','c','d','e','f','g']]
x_test=test[['y','z','a']]#'b','c','d','e','f','g']]
y_test=np.array(test[['h']]).ravel()
#print(yt)
clf.fit(xt,yt)
yp=clf.predict(x_test)
#print(yp)print(y_test)
a=confusion_matrix(y_test, yp)
print(a)
#print(yp)
#x1=[[xt1],[xt2],[xt3]]
#print(x1[:3])