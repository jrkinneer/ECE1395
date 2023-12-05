import numpy as np
import cv2
from kmeans_multiple import kmeans_multiple
from tqdm import tqdm
def recolor_img(original_features_shape, down_sampled_shape, ids, centroids):
    output = np.zeros(original_features_shape)
    
    for i in range(ids.shape[0]):
        output[i] = centroids[ids[i][0]]
    return output.reshape((down_sampled_shape[0], down_sampled_shape[1], 3))

# file_paths = ["./input/im1.jpg", "./input/im2.jpg", "./input/im3.png"]

# K_all = [3, 5, 7]
# iters_all = [7, 13, 20]
# R_all = [5, 15, 25]

# for ind, path in enumerate(file_paths):
#     img = cv2.imread(path)
#     img = np.array(img, dtype=float)

#     new_shape = (100,100)
#     down_sampled = img[::img.shape[0]//new_shape[0], ::img.shape[1]//new_shape[1]]
#     features = np.reshape(down_sampled, ((down_sampled.shape[0] * down_sampled.shape[1]), 3))

#     for k in tqdm(K_all, "k in K_all"):
#         for iters in tqdm(iters_all, "iteration in iters_all", leave=False):
#             for R in tqdm(R_all, "R in R_all", leave=False):
#                 ids, centroids, ssd = kmeans_multiple(features, k, iters, R)
#                 output = recolor_img(features.shape, down_sampled.shape, ids, centroids).astype('uint8')
#                 cv2.imwrite("./output/img"+str(ind + 1)+"_K"+str(k)+"_iters"+str(iters)+"_R"+str(R)+".png", output)

img = cv2.imread("./input/im3.png")
img = np.array(img, dtype=float)

new_shape = (100,100)
down_sampled = img[::img.shape[0]//new_shape[0], ::img.shape[1]//new_shape[1]]
features = np.reshape(down_sampled, ((down_sampled.shape[0] * down_sampled.shape[1]), 3))

ids, centroids, ssd = kmeans_multiple(features, 25, 20, 25)
output = recolor_img(features.shape, down_sampled.shape, ids, centroids).astype('uint8')
cv2.imwrite("./output/img3_K25_iters20_R25.png", output)