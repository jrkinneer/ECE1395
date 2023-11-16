import numpy as np
import scipy
from predict import predict
from nnCost import nnCost
#0
data = scipy.io.loadmat("./input/HW7_Data.mat")

X = np.array(data['X'])
y = np.array(data['y'])

#confirm shapes okay
if (X.shape[0] == y.shape[0]):
    print("shapes match")
else:
    print("error, incorrect feature and label shapes")
    exit(1)
    
thetas = scipy.io.loadmat("./input/HW7_weights_2.mat")
THETA_1 = np.array(thetas['Theta1'])
THETA_2 = np.array(thetas['Theta2'])

# print(THETA_1.shape)
# print(THETA_2.shape)

p, h_x = predict(THETA_1, THETA_2, X)

#1b accuracy measurement
count = 0

for i in range(p.shape[0]):
    if p[i][0] == y[i][0]:
        count += 1
        
accuracy = count/p.shape[0]
print("accuracy = ", accuracy)

#2a
lambdas = [0,1,2]
for lambda_ in lambdas:
    cost = nnCost(THETA_1, THETA_2, X, y, 3, lambda_)
    print("the cost for lambda (", lambda_,") is ", cost)