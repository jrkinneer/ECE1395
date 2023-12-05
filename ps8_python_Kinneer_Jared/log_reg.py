from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import scipy

def log_reg():
    data = scipy.io.loadmat("./input/subsets.mat")

    X3 = np.array(data["X3"])
    y3 = np.array(data["y3"])

    log_reg = LogisticRegression()
    log_reg.fit(X3, y3)

    error_list = []
    #classification errors for bagged X
    for i in range(5):
        bag_x = "X"+str(i+1)
        bag_y = "y"+str(i+1)
        
        X_temp = np.array(data[bag_x])
        y_temp = np.array(data[bag_y])
        
        y_pred = log_reg.predict(X_temp)
        
        classification_error = 1 - accuracy_score(y_temp, y_pred)
        
        # print("classification error for ", bag_x, " = ", classification_error)
        
        error_list.append(classification_error)
        
    #classification error for testing set
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
        
    y_pred = log_reg.predict(X_test)
        
    classification_error = 1 - accuracy_score(y_test, y_pred)

    error_list.append(classification_error)

    # print("classification error for X_test = ", classification_error)

    return error_list
