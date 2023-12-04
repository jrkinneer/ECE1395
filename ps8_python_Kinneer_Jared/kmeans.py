import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def kmeans_single(X, K, iters):
    ids = np.zeros((X.shape[0], 1), dtype='uint8')
    means = np.zeros((K, X.shape[1]))
    ssd = 0
    
    #randomly intitialize means
    for k in range(K):
        for n in range(X.shape[1]):
            means[k][n] = np.random.uniform(np.min(X[:, n]), np.max(X[:,n]))
           
    #runs kmeans for i iterations 
    for i in tqdm(range(iters), "i in iters, kmeans_single", leave=False):
        for m in range(X.shape[0]):
            #find the ids of the m datapoints
            distances_to_each_mean = cdist([X[m]], means, 'euclidean')
            
            closest_mean_index = np.argmin(distances_to_each_mean)
            
            ids[m][0] = int(closest_mean_index)
         
        
        #get the sum of each feature for the K means   
        sum_per_feature = np.zeros((K, X.shape[1]))
        
        for m2 in range(X.shape[0]):
            current_id = ids[m2][0]
            
            for n in range(X.shape[1]):
                sum_per_feature[current_id][n] += X[m2][n]
                
        for k in range(sum_per_feature.shape[0]):
            #get the amount of points in the cluster
            count_of_points_at_Ki = np.sum((ids == k))
            
            #update the means array
            if (count_of_points_at_Ki > 0):
                means[k, :] = sum_per_feature[k, :] / count_of_points_at_Ki
                
        
    #ssd calculation
    for m in range(X.shape[0]):
        cluster_id = ids[m]
        
        squared_dist = np.sum((X[m] - means[cluster_id]) ** 2)
        
        ssd += squared_dist
               
    return ids, means, ssd

# test_array = np.array([[12, 5, 7],
#                        [17, 10, 8],
#                        [14, 7, 12],
#                        [10, 3, 3],
#                        [9, 2, 1],
#                        [7, 0, 0]])

# ids, means, _ = kmeans_single(test_array, 2, 2)
# print(ids)
# print(means)
