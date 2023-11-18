import numpy as np
from predict import sigmoid
from sigmoidGradient import sigmoidGradient

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_, alpha, MaxEpochs):
    Theta1 = np.random.uniform(-.15, .15, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-.15, .15, (num_labels, hidden_layer_size + 1))
    
    y = np.zeros((y_train.shape[0], num_labels))
    for i in range(y_train.shape[0]):
        y[i][y_train[i][0]-1] = 1
        
    for a in range(MaxEpochs):
        for q in range(X_train.shape[0]):
            #forward pass
            
            #input layer plus bias
            X_q = np.hstack((1,X_train[q]))
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
            
            #back propagation
            y_q = np.reshape(y[q], (num_labels, 1))
            
            sigma_3 = np.reshape(a_3, (num_labels, 1)) - y_q
            
            first_term = np.dot(Theta2.T, sigma_3)
            sigma_2 = np.dot(first_term, np.reshape(sigmoidGradient(z_2), (1, z_2.shape[0])))
            
            sigma_2 = sigma_2[1:,:]
            
            #theta 2 update
            #gradient
            delta_2 = np.dot(sigma_3, a_2.T)
            #feature columns
            Theta2[:, 1:] = Theta2[:, 1:] - (alpha * (delta_2[:, 1:] + (lambda_*Theta2[:,1:]) ) )
            #bias column
            Theta2[:, 0] = Theta2[:, 0] - delta_2[:, 0] 
            
            #theta 1 update
            #gradient
            delta_1 = np.dot(sigma_2.T, X_q[1:])
            #feature columns
            Theta1[:, 1:] = Theta1[:, 1:] - (alpha * (delta_1[:, 1:] + (lambda_*Theta1[:, 1:]) ) )
            #bias column
            Theta1[:, 0] = Theta1[:, 0] - delta_2[:, 0]
            
            
    return Theta1, Theta2
    
def forward_pass(Theta1, Theta2, X):
    '''
    same function as predict except it returns z_2 as well as a_3
    '''
    #input layer plus bias
    X_q = np.hstack((1,X))
        
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
        
    return z_2, a_3