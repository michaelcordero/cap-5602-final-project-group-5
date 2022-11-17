#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first let's capture the data
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('weatherAUS.csv')
data['Date'] = pd.to_datetime(data["Date"])
data = data.dropna()
data = data.sort_values(by='Date')
data.head()


# In[2]:


# format the data
df = pd.get_dummies(data,columns=['Location','WindGustDir','WindDir9am','WindDir3pm'])
# convert yes's -> 1's, and no's -> 0's
df = df.replace(to_replace=['Yes', 'No'], value=[1,0])
# remove columns Date & RainTomorrow. Date might not be relevant. RainTomorrow is the Y variable.
columns = df.columns.values.tolist()
columns.remove('Date')
columns.remove('RainTomorrow')
# assign X & Y's
X = df[columns]
Y = df['RainTomorrow']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)
# check uniform shape of data
print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')


# In[3]:


# feed the classifier neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlpc = MLPClassifier(hidden_layer_sizes=(20))
chistory = mlpc.fit(X_train, Y_train)
yc_prediction = mlpc.predict(X_test)
mlpc_accuracy = accuracy_score(Y_test, yc_prediction)
print(f'MLPC Accuracy: {mlpc_accuracy}')


# In[4]:


# feed the regressor neural network
from sklearn.neural_network import MLPRegressor
import numpy as np

mlpr = MLPRegressor(hidden_layer_sizes=(20), solver='adam', activation='logistic')
rhistory = mlpr.fit(X_train, Y_train)
yr_prediction: np.ndarray = mlpr.predict(X_test)
yrm_prediction = np.array(list(map(lambda y: 1 if y >= 0.5 else 0, yr_prediction)))
mlpr_accuracy = accuracy_score(Y_test, yrm_prediction)
print(f'MLPR Accuracy: {mlpr_accuracy}')


# In[5]:


# mlp with keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

mlpk = Sequential()
# mlpk.add(Dense(50, activation='relu'))
# mlpk.add(Dense(25, activation='relu'))
mlpk.add(Dense(1, activation='sigmoid'))

# compile and train model
mlpk.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='sgd', metrics=['accuracy'])
mlpk.fit(X_train, Y_train, epochs=50)
# evaluate the model
mlpk_loss, mlpk_accuracy = mlpk.evaluate(X_test, Y_test)
print(f'MLPK accuracy: {mlpk_accuracy}')

