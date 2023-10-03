import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LogisticRegression

data = scipy.io.loadmat("./input/hw4_data3.mat")
X_train = np.array(data["X_train"])
y_train = np.array(data["y_train"])

X_test = np.array(data["X_test"])
y_test = np.array(data["y_test"])

X0 = np.ones((X_train.shape[0], 1))
X_train = np.hstack((X0, X_train))

X0 = np.ones((X_test.shape[0], 1))
X_test = np.hstack((X0, X_test))

#determine the unique classes
un = np.unique(y_train[:,0])

#create binary classifiers for all unique classes
y_train_binary = np.zeros((y_train.shape[0], len(np.unique(y_train[:,0]))))
for i in range(len(y_train)):
    for j in range(y_train_binary.shape[1]):
        if y_train[i] == un[j]:
            y_train_binary[i][j] = 1
        
#recast as int
y_train_binary = y_train_binary.astype('uint8')
   
#create list of models trained on each binary classifier (one vs all)     
models = [None] * len(un)
for i in range(len(un)):
    models[i] = LogisticRegression(random_state=0).fit(X_train, y_train_binary[:,i])
 
#get testing accuracy
count = 0
for i in range(X_test.shape[0]):
    
    #make prediction for all unique classes
    predictions = [None] * len(un)
    
    for j in range(len(un)):
        p = models[j].predict_proba(np.reshape(X_test[i], (1,-1)))
        predictions[j] = p[0][1]
     
    #find most confidently predicted class   
    m = max(predictions)
    ind = predictions.index(m)
    
    #increase success count if the predicted class is a match to the actual
    if un[ind] == y_test[i]:
        count += 1
        
print("testing accuracy = ", count/y_test.shape[0])

count = 0
for i in range(X_train.shape[0]):
    
    predictions = [None] * len(un)
    
    for j in range(len(un)):
        p = models[j].predict_proba(np.reshape(X_train[i], (1,-1)))
        predictions[j] = p[0][1]
        
    m = max(predictions)
    ind = predictions.index(m)
    
    if un[ind] == y_train[i]:
        count += 1
        
print("training accuracy = ", count/y_train.shape[0])
