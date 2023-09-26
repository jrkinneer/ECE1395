#1e
import math
import numpy as np
import sigmoid

def costFunction(theta, X_train, y_train):
    m = len(X_train)
    cost = 0
    for i in range(m):
        sigma = 0
        for j in range(len(X_train[i])):
            sigma += theta[j]*X_train[i][j]
        h_theta = sigmoid.sigmoid(sigma)
        if y_train[i] == 0:
            if h_theta != 1:
                cost += -math.log(1-h_theta)
        else:
            if h_theta != 0:
                cost += -math.log(h_theta)
        
    return cost
