     

     #1. Write a program to predict given review is helpful or not
	


from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

ad =ADASYN()
df= pd.read_csv('Assignment 1 dataset.csv',sep=',')
np.reshape(df,(14026,17))
train,test= train_test_split(df,test_size=0.20)
	
"""....TRAIN DATA...."""

x_train=train[['Nouns','Adjectives','Verbs','Entropy','Difficult_Words','Fletch_RE',
	'Dale_RS','li_tags','Set_length','Stop_words','total_words','wrong_words','lex_diversity','one_letter','two_letter','longer_letter_words']]

"""....TEST DATA...."""

x_test=test[['Nouns','Adjectives','Verbs','Entropy','Difficult_Words','Fletch_RE',
'Dale_RS','li_tags','Set_length','Stop_words','total_words','wrong_words','lex_diversity','one_letter','two_letter','longer_letter_words']]
y_train=train['Helpful']
y_test=test['Helpful']
x_train,y_train=ad.fit_sample(x_train,y_train)
x_test,y_test=ad.fit_sample(x_test,y_test)
clf = svm.SVC()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("predicted y value")
print(y_pred)
print("confusion matrix is")
a=confusion_matrix(y_test, y_pred)
print(a)
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
print("PRICISION ......")
p=tp/(tp+fp)
print(p)
p=tp / (tp + fn)
print("RECALL")
print(p)





