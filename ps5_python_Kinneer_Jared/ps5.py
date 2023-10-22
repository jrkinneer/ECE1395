import scipy
import numpy as np
import weightedKNN as w
import cv2
import random
import os
import matplotlib.pyplot as plt
#1
#Weighted KNN
# data = scipy.io.loadmat("./input/hw4_data3.mat")
# X_train = np.array(data["X_train"])
# y_train = np.array(data["y_train"])

# X_test = np.array(data["X_test"])
# y_test = np.array(data["y_test"])

# X0 = np.ones((X_train.shape[0], 1))
# X_train = np.hstack((X0, X_train))

# X0 = np.ones((X_test.shape[0], 1))
# X_test = np.hstack((X0, X_test))

# y_predict = w.weigthedKNN(X_train, y_train, X_test, 3.2)

# correct = 0
# for i in range(y_predict.shape[0]):
#     if y_predict[i] == y_test[i]:
#         correct += 1
    
# print("accuracy = ", correct/y_predict.shape[0])

#2.1a
# preprocessdata
# for i in range(1, 41):
#     train = random.sample(range(1, 11), 8)
#     for j in range(len(train)):
#         new_image_name = "Person"+str(i)+"_imageNumber"+str(train[j])
#         path = "./input/all/s"+str(i)+"/"+str(train[j])+".pgm"
#         new_path = "./input/train/"+new_image_name+".png"
        
#         img = cv2.imread(path)
#         cv2.imwrite(new_path, img)
#     for k in range(1, 11):
#         if k in train:
#             continue
#         else:
#             path = "./input/all/s"+str(i)+"/"+str(k)+".pgm"
#             new_image_name = "Person"+str(i)+"_imageNumber"+str(k)
            
#             new_path = "./input/test/"+new_image_name+".png"
#             img = cv2.imread(path)
#             cv2.imwrite(new_path, img)
#     asdf = 0

#2a
#compute T (10304x1)
t = np.zeros((10304, 1))
for file in os.listdir("./input/train"):
    img = cv2.imread("./input/train/"+file, cv2.IMREAD_GRAYSCALE)
    new_column = np.reshape(img, ((img.shape[0] * img.shape[1]), 1))
    t = np.hstack((t, new_column))

t = np.delete(t, 0, 1) 
# cv2.imwrite("./output/ps5-1-a.png", t)

#2b
#compute m
m = np.zeros((10304, 1))
for i in range(t.shape[0]):
    m[i] = np.average(t[i])
    
m = np.reshape(m, (112, 92))
# cv2.imwrite("./output/ps5-2-1-b.png", m)

#2c
#covariance matrix
m = np.reshape(m, (10304, 1))
a = t - m    
#covariance = np.dot(a, np.transpose(a)).astype('uint8')
cov2 = np.dot(np.transpose(a), a).astype('uint8')

# cv2.imwrite("./output/ps5-2-1-c.png", covariance)

#2d
#eigen values
eigenvalues, eigenvectors = scipy.linalg.eig(cov2)
eig_asINT = eigenvalues.astype('uint8')
sorted_eig = np.sort(eig_asINT)
sorted_eig[::-1].sort()
total_eig = np.sum(sorted_eig)

eig_sum = 0
k = 0
k_list = []
v_k = []
needed_eig = np.array([])
while (1):
    k += 1
    k_list.append(k)
    eig_sum = np.sum(sorted_eig[:k])/total_eig
    v_k.append(eig_sum)
    if (eig_sum > .95):
        needed_eig = sorted_eig[:k]
        break
    

# plt.plot(k_list, v_k)
# plt.show()

#2e
#basis matrix
u = np.zeros((320, 1))

indices = []
for i in range(k):
    inList = False
    counter = 0
    while (inList == False):
        location = np.where(eig_asINT == needed_eig[i])[0][counter]
        if location in indices:
            counter += 1
        else:
            indices.append(location)
            inList = True
            
    column = np.reshape( np.array(eigenvectors[:, indices[i]]), (320, 1))
    
    u = np.hstack((u, column))
u = np.delete(u, 0, 1)
# eigen_face = np.reshape(u[:, 0], (16, 20))
# for i in range(1, 10):
#     eigen_face = np.hstack((eigen_face, np.reshape(u[:, i], (16, 20))))
#cv2.imwrite("./output/ps5-2-1-e.png", eigen_face*255)

#2.2a
u_t = np.transpose(u)
w = u_t * (np.reshape(t[:, 0], (len(t[:, 0]), 1)) - m)