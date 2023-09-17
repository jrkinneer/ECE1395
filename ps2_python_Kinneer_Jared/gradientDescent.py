import numpy as np
import random
def gradientDescent(x_train, y_train, alpha, iters):
    theta = [random.random(), random.random()]
    m = x_train.shape[0]
    
    #for i iterations
    for i in range(iters):
        
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

#x array [x0, x1]
x_train = np.array([[1,0],[1,2],[1,3],[1,4]])
y_train = np.array([4,8,10,12])
alpha = .001
iter = 15

theta = gradientDescent(x_train, y_train, alpha, iter)
print("theta = [",theta[0],",",theta[1],"]")
cost = computeCost(x_train, y_train, theta)
print("cost: ", cost)