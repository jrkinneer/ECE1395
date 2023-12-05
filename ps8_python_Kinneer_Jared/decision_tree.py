from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy

def decision_tree():
    data = scipy.io.loadmat("./input/subsets.mat")

    X4 = np.array(data["X4"])
    y4 = np.array(data["y4"])

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X4, y4)

    error_list = []
    #classification errors for bagged X
    for i in range(5):
        bag_x = "X"+str(i+1)
        bag_y = "y"+str(i+1)
        
        X_temp = np.array(data[bag_x])
        y_temp = np.array(data[bag_y])
        
        y_pred = decision_tree.predict(X_temp)
        
        classification_error = 1 - accuracy_score(y_temp, y_pred)
        
        # print("classification error for ", bag_x, " = ", classification_error)
        
        error_list.append(classification_error)
        
    #classification error for testing set
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])
        
    y_pred = decision_tree.predict(X_test)
        
    classification_error = 1 - accuracy_score(y_test, y_pred)

    error_list.append(classification_error)

    # print("classification error for X_test = ", classification_error)

    return error_list

decision_tree()