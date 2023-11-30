import numpy as np
import scipy
import sklearn as sk
import cv2

#1a 
new_size = (20,20)
mat = scipy.io.loadmat("./input/HW8_data1.mat")
X = np.array(mat['X'])

rand_rows = [np.random.randint(0, X.shape[0]) for r in range(25)]

number_img = np.zeros((new_size[0] * 5, new_size[1]*5))
ind = 0
for i in range(25):
    img = np.reshape(X[rand_rows[i]], new_size)
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    row = i//5
    col = i%5
    
    x = col*new_size[0]
    y = row*new_size[1]
    
    number_img[x:x+new_size[0], y:y+new_size[1] ] = img
    
# cv2.imshow("numbers", number_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("./output/ps8-1-a-1.png", number_img)

#1b
y = np.array(mat['y'])
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
train_indices = indices[:4500]
test_indices = indices[4500:]
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

#1c bagging
bagging_indices = np.arange(X_train.shape[0])
np.random.shuffle(bagging_indices)
X1 = X_train[:900]
X2 = X_train[900:1800]
X3 = X_train[1800:2700]
X4 = X_train[2700:3600]
X5 = X_train[3600:]

y1 = y_train[:900]
y2 = y_train[900:1800]
y3 = y_train[1800:2700]
y4 = y_train[2700:3600]
y5 = y_train[3600:]

save_Data = {"X1": X1, "X2": X2, "X3":X3, "X4":X4, "X5":X5, "y1":y1, "y2":y2, "y3":y3, "y4":y4, "y5":y5, "X_test":X_test, "y_test":y_test}
scipy.io.savemat("./input/subsets.mat", save_Data)