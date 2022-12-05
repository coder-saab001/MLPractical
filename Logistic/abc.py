import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv('train_X.csv')
Y_train = pd.read_csv('train_Y.csv')

X_test = pd.read_csv('test_X.csv')
Y_test = pd.read_csv('test_Y.csv')

X_train = X_train.drop("Id", axis = 1)
Y_train = Y_train.drop("Id", axis = 1)
X_test = X_test.drop("Id", axis = 1)
Y_test = Y_test.drop("Id", axis = 1)

X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(X, Y, learning_rate, iterations):
    m = X.shape[1]
    n = X.shape[0]
    W = np.zeros((n, 1))
    B = 0
    cost_list = []
    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)

        W = W - learning_rate*dW.T
        B = B - learning_rate*dB

        cost_list.append(cost)
        if(i%(iterations/10) == 0):
            print("Cost after ", i, " iterations is: ", cost)
    return W, B, cost_list

iterations = 1
learning_rate = 0.0015
W, B, cost_list = model(X_train, Y_train, learning_rate, iterations)

plt.plot(cost_list)
plt.show()

def accuracy(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    A = A > 0.5
    A = np.array(A, dtype = 'int64')
    acc = (1 - np.sum(np.absolute(A-Y))/Y.shape[1])*100
    print("Accuracy of the model is : ", round(acc, 2), "%")

accuracy(X_test, Y_test, W, B)