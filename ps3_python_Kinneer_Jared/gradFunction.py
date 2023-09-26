import math
import numpy as np
import sigmoid

def gradFunction(theta, X_train, y_train):
    final_theta = np.empty(theta.shape)
    m = len(X_train)
    
    for j in range(len(theta)): 
        for i in range(m):
            sigma = 0
            for a in range(len(X_train[i])):
                sigma += theta[a]*X_train[i][a]
            final_theta += (sigmoid.sigmoid(sigma) - y_train[i])*X_train[i][j]
            
    return final_theta/m