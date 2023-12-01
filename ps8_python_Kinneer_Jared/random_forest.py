from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy

def rf():
    data = scipy.io.loadmat("./input/subsets.mat")

    X5 = np.array(data["X5"])
    y5 = np.array(data["y5"])

    rf = RandomForestClassifier(n_estimators=60)
    rf.fit(X5, y5)

    error_list = []
    #classification errors for bagged X
    for i in range(5):
        bag_x = "X"+str(i+1)
        bag_y = "y"+str(i+1)
        
        X_temp = np.array(data[bag_x])
        y_temp = np.array(data[bag_y])
        
        y_pred = rf.predict(X_temp)
        
        classification_error = 1 - accuracy_score(y_temp, y_pred)
        
        # print("classification error for ", bag_x, " = ", classification_error)
        
        error_list.append(classification_error)
        
    #classification error for testing set
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
        
    y_pred = rf.predict(X_test)
        
    classification_error = 1 - accuracy_score(y_test, y_pred)

    error_list.append(classification_error)

    # print("classification error for X_test = ", classification_error)

    return error_list

rf()