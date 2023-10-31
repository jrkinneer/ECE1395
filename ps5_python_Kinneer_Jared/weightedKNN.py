import numpy as np
from scipy import stats as st
K = 9
def weigthedKNN(X_train, y_train, X_test, sigma):
    y_predict = np.zeros((X_test.shape[0], 1)).astype('uint8')
    classes = len(np.unique(y_train))
    
    for i in range(X_test.shape[0]):
        distances = np.zeros((X_train.shape[0], 1))
        for j in range(X_train.shape[0]):
            dist = 0
            for k in range(X_train.shape[1]):
                dist += (X_test[i][k] - X_train[j][k])**2
            distances[j] = dist
        sorted_distances = np.sort(distances, 0)
        
        closest_K = sorted_distances[:K]
        
        indices = [None] * K
        
        for z in range(K):
            inList = False
            counter = 0
            while (inList == False):
                location = np.where(distances == closest_K[z])[0][counter]
                
                if location in indices:
                    counter += 1
                else:
                    indices[z] = location
                    inList = True
        #calculate weights
        weights = [None] * K
        for a in range(K):
            weights[a] = np.exp(distances[indices[a]]/sigma**2)
            
        #vote
        votes = [0] * (classes+1)
        for a in range(K):
            class_neighbor = y_train[indices[a]][0]
            votes[class_neighbor] += weights[a] * 1
        
        #find highest vote and index
        max_vote_score = max(votes)
        predicted_class = 0
        for b in range(classes):
            if votes[b] == max_vote_score:
                predicted_class = b
        y_predict[i] = predicted_class
        
    return y_predict