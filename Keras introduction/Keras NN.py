# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:17:03 2021

@author: tomerhochman
"""


# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

df = pd.read_csv(r"C:/Users/1/OneDrive/Desktop/Projects/Keras introduction/diabetes.csv",sep = ',')

X = df.values[:,0:8]
Y = df.values[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))