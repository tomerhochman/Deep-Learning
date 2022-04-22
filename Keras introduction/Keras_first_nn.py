# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:17:03 2021

@author: tomerhochman
"""


# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:/Users/1/OneDrive/Desktop/Projects/Keras introduction/diabetes.csv",sep = ',')

X = df.values[:,0:8]
Y = df.values[:,8]

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y,validation_split=0.30, epochs=150, batch_size=10, verbose=1)
Loss, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: ', Loss)
    
predictions = model.predict_classes(X)
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))        
 
    
#print(history.history.keys()) 

# =============================================================================
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.grid()
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')    
# =============================================================================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
