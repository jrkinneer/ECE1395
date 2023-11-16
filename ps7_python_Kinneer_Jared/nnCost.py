import numpy as np
from predict import predict
def nnCost(Theta1, Theta2, X, y, K, lambda_):
    cost_term1 = 0
    
    M = X.shape[0]
    
    #get h_theta for all inputs
    _, h_theta = predict(Theta1, Theta2, X)
    #prep y variables
    y_k = np.zeros((y.shape[0], K))
    for i in range(y.shape[0]):
        y_k[i][y[i][0]-1] = 1
        
    for i in range(M):
        for k in range(K):
            cost_term1 += (y_k[i][k] * np.log(h_theta[i][k])) + ((1-y_k[i][k])*np.log(1-h_theta[i][k]))
            
    cost_term1 = cost_term1*(-1/M)
    
    #sum squared theta
    cost_term2 = 0
    for m in range(Theta1.shape[0]):
        for n in range(Theta1.shape[1] - 1):
            cost_term2 += Theta1[m][n]**2
    for m in range(Theta2.shape[0]):
        for n in range(Theta2.shape[1] - 1):
            cost_term2 += Theta2[m][n]**2
            
    cost_term2 = cost_term2 * (lambda_/(2*M))
      
    cost = cost_term1 + cost_term2
    
    return cost 