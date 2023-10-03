import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

def Reg_normalEqn(X_train, y_train, lambda_):
    X_transposed = np.transpose(X_train)
    y_product = np.dot(X_transposed, y_train)
    
    M = np.identity(X_train.shape[1])
    M[0][0] = 0
    
    X_product = np.dot(X_transposed, X_train) + lambda_*M
    inv = np.linalg.pinv(X_product)
    
    theta = np.dot(inv, y_product)
    
    return theta

def computeCost(x, y, theta):
    cost = 0
    #loop through length of input vectors
    m = x.shape[0]
    for i in range(m):
        x_i = x[i]
        h = x_i[0]*theta[0] + x_i[1]*theta[1]
        difference = (h - y[i])**2
        cost += difference
        
    cost = cost * (1/(2*m))
    return cost

data = scipy.io.loadmat("./input/hw4_data1.mat")
X = np.array(data["X_data"])

X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

y = np.array(data["y"])



lambda_ = np.array([0, .001, .003, .005, .007, .009, .012, .017])
costs = np.zeros((20, len(lambda_)))
testing_costs = np.zeros((20, len(lambda_)))

average_cost = []
avg_testing_cost = []

for k in range(20):
    for j in range(len(lambda_)):
            
        train_i = 0
        test_i = 0
        
        X_train = np.zeros((X.shape[0], X.shape[1]))
        y_train = np.zeros((y.shape[0], y.shape[1]))

        X_test = np.zeros((X.shape[0], X.shape[1]))
        y_test = np.zeros((y.shape[0], y.shape[1]))
        
        for i in range(X.shape[0]):
            r = random.random()
            if (r < .88):
                X_train[train_i] = X[i]
                y_train[train_i] = y[i]
                train_i += 1
            else:
                X_test[test_i] = X[i]
                y_test[test_i] = y[i]
                test_i += 1
                

        X_train = np.resize(X_train, (train_i, X.shape[1]))
        X_test = np.resize(X_test, (test_i, X.shape[1]))
        y_train = np.resize(y_train, (train_i, 1))
        y_test = np.resize(y_test, (test_i, 1))

        theta = Reg_normalEqn(X_train, y_train, lambda_[j])
        costs[k][j] = computeCost(X_train, y_train, theta)
        testing_costs[k][j] = computeCost(X_test, y_test, theta)
    
for i in range(len(lambda_)):
    if (i != 0):
        average_cost.append(np.sum(costs[:,i])/costs.shape[0])
        avg_testing_cost.append(np.sum(testing_costs[:,i])/testing_costs.shape[0])

plotting_lambda = [.001, .003, .005, .007, .009, .012, .017]
plt.plot(plotting_lambda, average_cost, label="training cost")
plt.plot(plotting_lambda, avg_testing_cost, label="testing cost")
plt.xlabel("lambda")
plt.ylabel("average cost")
plt.legend()
plt.show()
