from kmeans import kmeans_single
import numpy as np
from tqdm import tqdm

def kmeans_multiple(X, K, iters, R):
    all_ids = np.zeros((X.shape[0], R), dtype='uint8')
    
    all_centroids = np.zeros((K, X.shape[1], R))
    
    all_ssd = [0]*R
    
    for r in tqdm(range(R), "r in kmeans_multiple", leave=False):
        id_r, mean_r, ssd_r = kmeans_single(X, K, iters)
        
        all_ids[:, r] = np.reshape(id_r, (X.shape[0],))
        all_centroids[:,:,r] = mean_r
        all_ssd[r] = ssd_r
        
    best_r = all_ssd.index(min(all_ssd))
    
    return np.reshape(all_ids[:, best_r], (X.shape[0], 1)), all_centroids[:,:,best_r], all_ssd[best_r]
    
# test_array = np.array([[12, 5, 7],
#                        [17, 10, 8],
#                        [14, 7, 12],
#                        [10, 3, 3],
#                        [9, 2, 1],
#                        [7, 0, 0]])

# ids, means, _ = kmeans_multiple(test_array, 2, 2, 2)
# print(ids)
# print(means)