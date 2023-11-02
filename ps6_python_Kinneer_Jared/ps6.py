import numpy as np
import scipy
import math as m

data = scipy.io.loadmat("./input/hw4_data3.mat")
X_train = np.array(data["X_train"])
y_train = np.array(data["y_train"])
X_test = np.array(data["X_test"])
y_test = np.array(data["y_test"])


X_train_1 = np.zeros((1, X_train.shape[1]))
X_train_2 = np.zeros((1, X_train.shape[1]))
X_train_3 = np.zeros((1, X_train.shape[1]))

for i in range(y_train.shape[0]):
    match y_train[i]:
        case 1:
            X_train_1 = np.vstack((X_train_1, X_train[i]))
        case 2:
            X_train_2 = np.vstack((X_train_2, X_train[i]))
        case 3:
            X_train_3 = np.vstack((X_train_3, X_train[i]))
     
np.delete(X_train_1, 0, 0)       
np.delete(X_train_2, 0, 0)       
np.delete(X_train_3, 0, 0)      
# print("size of X_train_1 = ", X_train_1.shape[0])
# print("size of X_train_2 = ", X_train_2.shape[0])
# print("size of X_train_3 = ", X_train_3.shape[0])

class1_mean = np.mean(X_train_1, 0)
class2_mean = np.mean(X_train_2, 0)
class3_mean = np.mean(X_train_3, 0)

class1_stddev = np.std(X_train_1, 0)
class2_stddev = np.std(X_train_2, 0)
class3_stddev = np.std(X_train_3, 0)

# print("class, feature 1 mean, feature 2 mean, feature 3 mean, feature 4 mean")
# print("1", class1_mean)
# print("2", class2_mean)
# print("3", class3_mean)

# print("class, feature 1 stddev, feature 2 stddev, feature 3 stddev, feature 4 stddev")
# print("1", class1_stddev)
# print("2", class2_stddev)
# print("3", class3_stddev)

#b
accuracy = 0
for i in range(X_test.shape[0]):
    #bi
    pj_w1 = [None]*X_test.shape[1]
    pj_w2 = [None]*X_test.shape[1]
    pj_w3 = [None]*X_test.shape[1]
    
    for j in range(X_test.shape[1]):
        pj_w1[j] = (1/(m.sqrt(2*3.14)*class1_stddev[j]))*m.exp(-((X_test[i][j] - class1_mean[j])**2)/(2*(class1_stddev[j]**2)))
        pj_w2[j] = (1/(m.sqrt(2*3.14)*class2_stddev[j]))*m.exp(-((X_test[i][j] - class2_mean[j])**2)/(2*(class2_stddev[j]**2)))
        pj_w3[j] = (1/(m.sqrt(2*3.14)*class3_stddev[j]))*m.exp(-((X_test[i][j] - class3_mean[j])**2)/(2*(class3_stddev[j]**2)))
    ################################
    
    #bii  
    ln_wi = [0] * 3
    for k in range(X_test.shape[1]):
        ln_wi[0] += m.log(pj_w1[k])
        ln_wi[1] += m.log(pj_w2[k])
        ln_wi[2] += m.log(pj_w3[k])
    ################################   
      
    #biii  
    ln_w_x = [0]*3
    for a in range(3):
        ln_w_x[a] = ln_wi[a] + m.log(1/3)
    #####################################
    
    #biv    
    most_probable = ln_w_x.index(max(ln_w_x)) + 1
    
    #bv
    if most_probable == y_test[i]:
        accuracy+=1
        
accuracy= accuracy/X_test.shape[0]

# print("accuracy = ", accuracy)

#2
sigma_1 = np.cov(X_train_1)
sigma_2 = np.cov(X_train_2)
sigma_3 = np.cov(X_train_3)

# print(sigma_1.shape)
# print(sigma_2.shape)
# print(sigma_3.shape)

# print(sigma_1)

#2b calculated above
# print(class1_mean.shape)
# print(class2_mean.shape)
# print(class3_mean.shape)

#2c
def discrim(feature, mean, std_dev):
    exp = -((feature - mean)**2)/(2*(std_dev**2))
    log_l = exp - m.log(std_dev) - (.5*m.log(2*m.pi))
    return log_l
   
accuracy = 0 
for i in range(X_test.shape[0]):
    g = [0]*3
    for j in range(X_test.shape[1]):
        g[0] += discrim(X_test[i][j], class1_mean[j], class1_stddev[j])
        g[1] += discrim(X_test[i][j], class2_mean[j], class2_stddev[j])
        g[2] += discrim(X_test[i][j], class3_mean[j], class3_stddev[j])
        
    most_probable = g.index(max(g)) + 1
    
    if most_probable == y_test[i]:
        accuracy += 1
        
accuracy = accuracy/y_test.shape[0]

print("accuracy = ", accuracy)