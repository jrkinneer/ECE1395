from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy

def knn():
    data = scipy.io.loadmat("./input/subsets.mat")

    X2 = np.array(data["X2"])
    y2 = np.array(data["y2"])

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X2, y2)

    error_list = []
    #classification errors for bagged X
    for i in range(5):
        bag_x = "X"+str(i+1)
        bag_y = "y"+str(i+1)
        
        X_temp = np.array(data[bag_x])
        y_temp = np.array(data[bag_y])
        
        y_pred = knn.predict(X_temp)
        
        classification_error = 1 - accuracy_score(y_temp, y_pred)
        
        # print("classification error for ", bag_x, " = ", classification_error)
        
        error_list.append(classification_error)
        
    #classification error for testing set
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
        
    y_pred = knn.predict(X_test)
        
    classification_error = 1 - accuracy_score(y_test, y_pred)

    error_list.append(classification_error)

    # print("classification error for X_test = ", classification_error)

    return error_list
