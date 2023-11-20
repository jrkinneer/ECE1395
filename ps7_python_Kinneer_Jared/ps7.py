import numpy as np
import scipy
from predict import predict
from nnCost import nnCost
from sigmoidGradient import sigmoidGradient
from sGD import sGD

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
print("accuracy = ", accuracy,"\n")

#2a,b
lambdas = [0,1,2]
for lambda_ in lambdas:
    cost = nnCost(THETA_1, THETA_2, X, y, 3, lambda_)
    print("the cost for lambda (", lambda_,") is ", cost)
    
#3
test_z = np.array([-10,0,10])
print("\ntest of sigmoid gradient with z = [-10,0,10]\ng'(z)=",sigmoidGradient(test_z))

#4
alpha = .1
# X_train = np.zeros_like(X)
# y_train = np.zeros_like(y)
# X_test = np.zeros_like(X)
# y_test = np.zeros_like(y)
# counter = 0
# counter_test = 0
# for i in range(X.shape[0]):
#     r = np.random.rand()
#     if r < .85:
#         X_train[counter] = X[i]
#         y_train[counter] = y[i]
#         counter+=1
#     else:
#         X_test[counter_test] = X[i]
#         y_test[counter_test] = y[i]
#         counter_test += 1 
        
# X_train = np.resize(X_train, (counter, X_train.shape[1]))
# y_train = np.resize(y_train, (counter, y_train.shape[1]))

# THETA_1_trained, THETA_2_trained = sGD(4, 8, 3, X_train, y_train, .01, alpha, 3)
#5

Epochs = [50, 100]
lambdas = [0,.01,.1,1]

for epoch in Epochs:
    for lambda_x in lambdas:
        #randomize testing and training data
        X_train = np.zeros_like(X)
        y_train = np.zeros_like(y)
        X_test = np.zeros_like(X)
        y_test = np.zeros_like(y)
        counter = 0
        counter_test = 0
        for i in range(X.shape[0]):
            r = np.random.rand()
            if r < .85:
                X_train[counter] = X[i]
                y_train[counter] = y[i]
                counter+=1
            else:
                X_test[counter_test] = X[i]
                y_test[counter_test] = y[i]
                counter_test += 1 
                
        X_train = np.resize(X_train, (counter, X_train.shape[1]))
        y_train = np.resize(y_train, (counter, y_train.shape[1]))
        X_test = np.resize(X_test, (counter_test, X.shape[1]))
        y_test = np.resize(y_test, (counter_test, y.shape[1]))
        
        #use training data to create theta's
        THETA_1_trained, THETA_2_trained = sGD(4,8,3,X_train, y_train, lambda_x, alpha, epoch)
        
        #use theta's to predict output
        p_train, h_x_train = predict(THETA_1_trained, THETA_2_trained, X_train)
        p_test, h_x_test = predict(THETA_1_trained, THETA_2_trained, X_test)
        
        #get accuracy for training and testing data
        train_accuracy = 0
        test_accuracy = 0
        
        for i in range(p_train.shape[0]):
            if p_train[i][0] == y_train[i][0]:
                train_accuracy += 1
                
        for i in range(p_test.shape[0]):
            if p_test[i][0] == y_test[i][0]:
                test_accuracy += 1
                
        train_accuracy /= p_train.shape[0]
        test_accuracy /= p_test.shape[0]
        
        train_cost = nnCost(THETA_1_trained, THETA_2_trained, X_train, y_train, 3, lambda_x)
        test_cost = nnCost(THETA_1_trained, THETA_2_trained, X_test, y_test, 3, lambda_x)
        
        print("for MaxEpochs = ", epoch, " lambda = ", lambda_x, " alpha = ", alpha, " training data accuracy = ", train_accuracy * 100, "%", " cost = ", train_cost)
        print("for MaxEpochs = ", epoch, " lambda = ", lambda_x, " alpha = ", alpha, " testing data accuracy = ", test_accuracy * 100, "%", " cost = ", test_cost)
        print("\n")
        