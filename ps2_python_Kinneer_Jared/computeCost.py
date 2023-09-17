import numpy as np

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
x = np.array([[1,0],[1,2],[1,3],[1,4]])
y = np.array([4,8,10,12])
theta = np.array([1, 1])
cost = computeCost(x, y, theta)
print(cost)