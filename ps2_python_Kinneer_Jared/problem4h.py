import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

#plots lines on the current active plot
def abline(slope, intercept, linetype='--', l='', linew=1):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linetype, label=l, linewidth=linew)

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

def gradientDescent(x_train, y_train, alpha, iters, cost_array, iteration_array):
    theta = [random.random(), random.random()]
    m = x_train.shape[0]
    
    #for i iterations
    for k in range(iters):
        
        #find derivative cost of theta0
        cost0 = 0
        for i in range(m):
            x_i = x_train[i]
            h = x_i[0]*theta[0] + x_i[1]*theta[1]
            difference = (h - y_train[i])
            cost0 += difference
        cost0 = cost0 * (alpha/m)
        
        #find derivative cost of theta1
        cost1 = 0
        for i in range(m):
            x_i = x_train[i]
            h = x_i[0]*theta[0] + x_i[1]*theta[1]
            cost1 += (h - y_train[i]) * x_i[1]
            
        cost1 = cost1 * (alpha/m)
        
        #increment theta
        temp0 = theta[0] - cost0
        temp1 = theta[1] - cost1
        theta = [temp0, temp1]
        
        cost_array.append(computeCost(x_train, y_train, theta))
        iteration_array.append(k)
        
        #graphs line based on what iteration we are on
        if (k == iters-1):
            abline(theta[1], theta[0], 'r-', l="final fit of theta", linew=5)
        elif (k == 0):
            abline(theta[1], theta[0], l="initial fit of theta", linew=3)
        else:
            abline(theta[1], theta[0])
                
    return theta

data = pd.read_csv("./input/hw2_data1.csv")
rows = len(data)
X = np.empty((rows + 1, 2))
x1_only = np.empty(rows + 1)
y = np.empty(rows + 1)

#load data into arrays
with open("./input/hw2_data1.csv") as file:
    i = 0
    for row in file:
        x1 = row.partition(',')[0]
        
        X[i][0] = 1
        X[i][1] = x1
        x1_only[i] = x1
        y[i] = row.partition(',')[2]
        
        i += 1
        
plt.scatter(x1_only, y)
plt.xlim([0,2.6])
plt.ylim([0,50])

#split data into training and test set
X_train = np.empty(shape=[0,2])
X_test = np.empty(shape=[0,2])
y_train = []
y_test = []

for i in range(rows + 1):
    r = random.random()
    if (r < .9):
       X_train = np.append(X_train, [[X[i][0], X[i][1]]], axis=0)
       y_train.append(y[i])
    else: 
       X_test = np.append(X_test, [[X[i][0], X[i][1]]], axis=0)
       y_test.append(y[i])

#calculate theta
lrs = [.001, .003, .03, 3]
for a in range(4):
    cost_array = []
    iteration_array = []
    theta = gradientDescent(X_train, y_train, lrs[a], 300, cost_array, iteration_array)
    print("theta = [",theta[0],",",theta[1],"]")
    plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.scatter(iteration_array, cost_array)
    s = "iteration vs cost for alpha="+str(lrs[a])
    plt.title(s)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()

    #use test data to compute error
    total_cost = 0
    for k in range(len(X_test)):
        y_pred = (X_test[k][0]*theta[0]) + (X_test[k][1]*theta[1])
        cost_i = (y_pred - y_test[k])**2
        total_cost += cost_i
    print("total cost after testing gradient descent: ", total_cost/(2*len(X_test)))

