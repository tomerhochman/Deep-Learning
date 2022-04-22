# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:10:43 2021

@author: Tomer Hochman
"""

# =================================== Ex 1 ====================================

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function

def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

# Sigmoid Derivative

def sigmoid_deriv(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))


# input data set
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output data set
Y = np.array([[0],[0],[1],[1]])

# Initialize random values for all W ( the 2* and -1  means we expand our range of values to be between -1 and 1) 

W = 2*np.random.random((3,1)) -1 


Epochs = 1000
m = len(Y)
Loss_lst = []
lr = 1


for i in range(Epochs):
    
    # forward pass step 1
    Z = X
    
    # forward pass step 2
    a = sigmoid_func(np.dot(X,W))
    
    # backpropagation step 1
    Loss = (1 /(2*m)) * (np.sum(np.power((a - Y), 2)))
    Loss_lst.append(Loss)
    
    Error = a - Y
    
    # backpropagation step 2
    z_delta = Error * sigmoid_deriv(a)
    
    W = W - lr * np.dot(X.T,z_delta)
    
    
print("the Predicted Y: \n", a,  "\n\n" , "the weights W: \n" ,W)
  
    
# plotting Loss
plt.plot(Loss_lst)
plt.ylabel('Loss')
plt.xlabel("Epochs:")

print(" Loss \n",Loss)

############################### EX 2 #########################################

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function

def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

# Sigmoid Derivative

def sigmoid_deriv(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))


# input data set
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output data set
Y = np.array([[0],[0],[1],[1]])

# Initialize random values for all W ( the 2* and -1  means we expand our range of values to be between -1 and 1) 

W1 = 2*np.random.random((3,4)) -1 
W2 = 2*np.random.random((4,1)) -1

Epochs = 2000
m = len(Y)
Loss_lst = []
lr = 1


for i in range(Epochs):
    
    # forward pass 
    Z1 = np.dot(X,W1)
    a1 = sigmoid_func(Z1)
    Z2 = np.dot(a1,W2)
    y_predict = sigmoid_func(Z2)
     
    # Backpropagation step 1
    Loss = ((1 / (2*m)) * (np.sum(np.power((y_predict - Y), 2))))
    Loss_lst.append(Loss)
    
    # Backpropagation step 2 - from output layer backward to input layer
    Error2 = y_predict - Y    
    L2_d = Error2 * sigmoid_deriv(y_predict)
    
    Error1 = np.dot(L2_d,W2.T)
    L1_d = Error1 * sigmoid_deriv(a1)
    
    # Update the weights
    W2 = W2 - lr * np.dot(a1.T,L2_d)
    W1 = W1 - lr * np.dot(X.T,L1_d)
    
    
print("the Predicted Y: \n", y_predict, "\n\n")
print(" Loss \n",Loss, "\n\n")
print("W2: \n",W2,"\n\n" )
print("W1: \n",W1)
   
# plotting Loss
plt.plot(Loss_lst)
plt.ylabel('Loss')
plt.xlabel("Epochs:")





