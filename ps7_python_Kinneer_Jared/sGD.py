import numpy as np
from predict import sigmoid
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost
import matplotlib.pyplot as plt

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_, alpha, MaxEpochs):
    Theta1 = np.random.uniform(-.15, .15, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-.15, .15, (num_labels, hidden_layer_size + 1))
    
    y = np.zeros((y_train.shape[0], num_labels))
    for i in range(y_train.shape[0]):
        y[i][y_train[i][0]-1] = 1
        
    training_costs = []
    epochs = []
    
    for a in range(MaxEpochs):
        for q in range(X_train.shape[0]):
            #forward pass
            #*****************************************************************
            #input layer plus bias
            X_q = np.hstack((1,X_train[q]))
            #input to hidden layer
            z_2 = np.dot(Theta1, X_q)
                
            #hidden layer values
            a_2 = sigmoid(z_2)
            #add bias before output layer
            a_2 = np.hstack((1, a_2))
            a_2 = np.reshape(a_2, (a_2.shape[0], 1))
                
            #inputs to output layer
            z_3 = np.dot(Theta2, a_2)
                
            #value of output layer
            a_3 = sigmoid(z_3)
            a_3 = np.reshape(a_3, (a_3.shape[0], 1))
            #*****************************************************************
            
            #back propagation
            #*****************************************************************
            y_q = np.reshape(y[q], (num_labels, 1))
            
            sigma_3 = np.reshape(a_3, (num_labels, 1)) - y_q
            
            # first_term = np.dot(Theta2.T, sigma_3)
            # grad_result = sigmoidGradient(z_2)
            # grad_result = np.reshape(grad_result, (1, z_2.shape[0]))
            # sigma_2 = np.dot(first_term, grad_result)
            
            # sigma_2 = np.reshape(sigma_2[1:,:], (hidden_layer_size, 1))
            sigma_2 = np.dot(Theta2.T, sigma_3)
            grad_result = np.reshape(sigmoidGradient(z_2), (z_2.shape[0], 1))
            sigma_2 = sigma_2[1:, :] * grad_result
            #*****************************************************************
            
            #theta 2 update
            #gradient
            delta_2 = np.dot(sigma_3, a_2.T)
            #feature columns
            Theta2[:, 1:] = Theta2[:, 1:] - (alpha * (delta_2[:, 1:] + (lambda_*Theta2[:,1:]) ) )
            #bias column
            Theta2[:, 0] = Theta2[:, 0] - delta_2[:, 0] 
            
            #theta 1 update
            #gradient
            delta_1 = np.dot(sigma_2, X_q[np.newaxis, :])
            #feature columns
            Theta1[:, 1:] = Theta1[:, 1:] - (alpha * (delta_1[:, 1:] + (lambda_*Theta1[:, 1:]) ) )
            #bias column
            Theta1[:, 0] = Theta1[:, 0] - delta_1[:, 0]
            
        #calculate training cost
        # cost = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda_)
        # training_costs.append(cost)
        # epochs.append(a+1)
        
    # plt.plot(epochs, training_costs)
    # plt.xlabel("epoch")
    # plt.ylabel("training cost")
    # plt.xticks(epochs)
    # plt.show()
            
    return Theta1, Theta2
