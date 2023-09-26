import numpy as np
import matplotlib.pyplot as plt
import csv

def normal_equation(X, y):
    X_transpose = np.transpose(X)
    X_product = np.dot(X_transpose, X)
    y_product = np.dot(X_transpose, y)
    
    theta = np.linalg.solve(X_product, y_product)
    
    return theta

with open("./input/hw3_data2.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = np.array(data)
Y = np.empty(len(data))
for i in range(len(data)):
    Y[i] = float(data[i][1])/(10**6)
    
plt.scatter(np.asarray(data[:,0], dtype='float'), Y , color='orange', marker='x', label='Training Data')

X = np.ones((len(data), 3))
X[:,1] = np.asarray(data[:,0])
for i in range(len(data)):
    X[i][2] = X[i][1]**2

theta = normal_equation(X, Y)
print("theta: ", theta)

temp_x = np.linspace(np.min(X, axis=0)[1], np.max(X, axis=0)[1], 100)
fitted_line = theta[0] + temp_x*theta[1] + ((temp_x**2) * theta[2])
plt.plot(temp_x, fitted_line, label='fitted model')
plt.xlabel("population in thousands")
plt.ylabel("profit in millions")
plt.legend()
plt.show()

asdf = 0