import numpy as np

def predict(Theta1, Theta2, X):
    
    h_theta = np.zeros((X.shape[0], 3))
    p = np.zeros((X.shape[0], 1))
    
    for q in range(X.shape[0]):
        #input layer plus bias
        X_q = np.hstack((1,X[q]))
        
        #input to hidden layer
        z_2 = np.dot(Theta1, X_q)
        
        #hidden layer values
        a_2 = sigmoid(z_2)
        #add bias before output layer
        a_2 = np.hstack((1, a_2))
        
        #inputs to output layer
        z_3 = np.dot(Theta2, a_2)
        
        #value of output layer
        a_3 = sigmoid(z_3)
        
        #add to return values
        h_theta[q] = a_3
        p[q][0] = np.argmax(a_3) + 1
        
    return p, h_theta
        
def sigmoid(z):
    return 1/(1+np.exp(-z))