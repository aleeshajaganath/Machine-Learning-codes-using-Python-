from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Activation
# fix random seed for reproducibility
import pandas as p
import numpy as np
from sklearn.decomposition import PCA
from keras.utils import np_utils

df = p.read_csv("iris.csv")
df.sample(frac=1)
from sklearn.cross_validation import train_test_split
train, test = train_test_split(df, test_size=0.2)

x_train = train[['sepalwidth','sepallength','petallength','petalwidth']]  
y_train=train['class']
from keras.utils import to_categorical
y_train = to_categorical(y_train)

x_test = test[['sepallength','sepalwidth','petallength','petalwidth']]  
y_test=test['class']
from keras.utils import to_categorical
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(6,input_dim=4))
model.add(Activation("relu"))
model.add(Dense(output_dim=3))
model.add(Activation("softmax"))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(x_train, y_train, epochs=100, batch_size=1)

# evaluate the model
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
