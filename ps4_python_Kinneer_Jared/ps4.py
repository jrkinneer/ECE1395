import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier

#effect of K

data = scipy.io.loadmat("./input/hw4_data2.mat")
K = np.linspace(1, 15, num=8)
counts = np.zeros((5,8))

labels = np.array(["X1", "X2", "X3", "X4", "X5"])
labels_y = np.array(["y1", "y2", "y3", "y4", "y5"])

#create the five folds
for i in range(5):
    use_labels = labels[np.arange(len(labels)) != i]
    use_labels_y = labels_y[np.arange(len(labels_y)) != i]
    
    X_train2 = data[use_labels[len(use_labels) - 1]]
    y_train2 = data[use_labels_y[len(use_labels_y) - 1]]
    
    X_test = data[labels[i]]
    y_test = data[labels_y[i]]
    
    j = len(use_labels) - 2
    
    while j > -1:
        X_temp = data[use_labels[j]]
        X_train2 = np.vstack((X_train2, X_temp))
        
        y_temp = data[use_labels_y[j]]
        y_train2 = np.vstack((y_train2, y_temp))
        
        j -= 1
        
    #add X0 to training and testing 
    X0 = np.ones((X_train2.shape[0], 1))
    X_train2 = np.hstack((X0, X_train2))
    
    X0 = np.ones((X_test.shape[0], 1))
    X_test = np.hstack((X0, X_test))
    
    v = 0
    for vals in K:
        n = KNeighborsClassifier(n_neighbors=int(vals))
        n.fit(X_train2, y_train2.ravel())
        
        correct_count = 0
        for a in range(X_test.shape[0]):
            y_hat = n.predict(np.reshape(X_test[a], (1,-1)))
            if y_hat == y_test[a]:
                correct_count += 1
        counts[i][v] = correct_count
        v += 1
avg_counts = []
for i in range(len(K)):
    avg_counts.append(np.sum(counts[:,i])/counts.shape[0])
plt.plot(K, avg_counts)
plt.xlabel("K")
plt.ylabel("average # correct guesses")
plt.show()