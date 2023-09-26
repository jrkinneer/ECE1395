import numpy as np
import matplotlib.pyplot as plt
import costFunction
import scipy
import random

def abline(slope, intercept, X_train):
    """Plot a line from slope and intercept"""
    m = len(X_train)
    x_vals = np.linspace(min(np.min(X_train, axis=0)[1], np.min(X_train, axis=0)[2]), max(np.max(X_train, axis=0)[1], np.max(X_train, axis=0)[2]), 100)
    y_vals = (intercept + slope * x_vals)
    plt.plot(x_vals, y_vals, '--')

#1a
#load data and get length
temp_X = np.loadtxt("./input/hw3_data1.txt", usecols=[0,1], dtype='float', delimiter=',')
Y = np.loadtxt("./input/hw3_data1.txt", usecols=2, dtype='int', delimiter=',')

X = np.empty((len(temp_X), 3))
for i in range(len(temp_X)):
    X[i] = [1, temp_X[i][0], temp_X[i][1]]

#1b
#plot data
X_admitted = np.empty((len(temp_X), 2))
X_not = np.empty((len(temp_X), 2))
admitted_len = 0
not_len = 0
for i in range(len(temp_X)):
    if Y[i] == 1:
        X_admitted[admitted_len] = [temp_X[i][0], temp_X[i][1]]
        admitted_len += 1
    else:
        X_not[not_len] = [temp_X[i][0], temp_X[i][1]]
        not_len += 1
        
X_admitted = np.resize(X_admitted, (admitted_len, 2))
X_not = np.resize(X_not, (not_len, 2))
print("not len: ", not_len, " admitted len: ", admitted_len)

plt.scatter(X_admitted[:,0], X_admitted[:,1], color='blue', marker='o', label='Admitted')
plt.scatter(X_not[:,0], X_not[:,1], color='red', marker='x', label='Not Admitted')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()

#1c 
#divide into training and test
X_test = np.empty((len(temp_X), 3))
X_train = np.empty((len(temp_X), 3))
Y_test = np.empty((len(temp_X), 1))
y_train = np.empty((len(temp_X), 1))
train_len = 0
test_len = 0
for i in range(len(temp_X)):
    r = random.random()
    if r < .9:
        X_train[train_len] = [1, temp_X[i][0], temp_X[i][1]]
        y_train[train_len] = [Y[i]]
        train_len += 1
    else:
        X_test[test_len] = [1, temp_X[i][0], temp_X[i][1]]
        Y_test[test_len] = [Y[i]]
        test_len += 1
        
X_test = np.resize(X_test, (test_len, 3))
Y_test = np.resize(Y_test, (test_len, 1))
X_train = np.resize(X_train, (train_len, 3))
y_train = np.resize(y_train, (train_len, 1))

#1f
initial_theta = np.zeros((3))
print(initial_theta)
final_theta = scipy.optimize.fmin_bfgs(costFunction.costFunction, initial_theta,args=(X_train, y_train))
print("initial theta: ", initial_theta)
print("cost with initial theta: ", costFunction.costFunction(initial_theta, X_train, y_train))
print("final theta: ", final_theta)
print("cost with final theta: ", costFunction.costFunction(final_theta, X_train, y_train))
#1g plot classification line
abline(-final_theta[1]/final_theta[2], -final_theta[0]/final_theta[2], X_train)

plt.show()

asdf = 0