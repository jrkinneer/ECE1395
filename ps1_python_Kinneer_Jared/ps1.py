import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
#3a
x = .6*np.random.randn(1000000, 1) + 1.5
#3b
z = (np.random.rand(1000000, 1) * (4))+ -1
#3c
nbins = 100
fig, ax = plt.subplots()
ax.hist(x, bins = nbins)
fig2, ax2 = plt.subplots()
ax2.hist(z, bins = nbins)

#3d
x2 = x
length = x.shape[0]
start = time.time()
for i in range(length):
    x2[i][0] = x[i][0] + 1
    
finish = time.time()
print("elapsed time: ", finish-start)

#3e
add = np.ones((1000000, 1))
start = time.time()
sumX1 = x + add
finish = time.time()
print("elapsed time vector addition: ", finish-start)

#3f
count = 0
for i in range(length):
    if (z[i][0] < 1.5):
        count += 1
print("count= ", count)

#4a
A = [[2,1,3], [2,6,8], [6,8,18]]
df = pd.DataFrame(A)
numA = np.array(A)

minCol0 = df.min(axis='columns')[0]
minCol1 = df.min(axis='columns')[1]
minCol2 = df.min(axis='columns')[2]

maxRow0 = df.max(1)[0]
maxRow1 = df.max(1)[1]
maxRow2 = df.max(1)[2]

minA = np.min(numA)

sumR0 = np.sum(numA, 1)[0]
sumR1 = np.sum(numA, 1)[1]
sumR2 = np.sum(numA, 1)[2]

sumAll = np.sum(numA)

B = np.square(numA)

#4b
sysA = np.array([[2,1,3], [2,6,8], [6,8,18]])
b = np.array([[1],[3],[5]])
#x = [[x],[y],[z]]
x = np.dot(np.linalg.inv(sysA) , b) 

#4c
normX1 = np.linalg.norm(np.array([.5,0,-1.5]))
normX2 = np.linalg.norm(np.array([1, -1 , 0]))
print(normX1, ' ', normX2)

#5
testA = np.array([[1,2,3], [4,5,6]])
testB = np.array([[1,2,3], [4,5,6] , [7,8,9]])
def sum_sq_row(array):
    cols = array.shape[1]
    rows = array.shape[0]
    sumCol = 0
    B = []
    for i in range(cols):
        for j in range(rows):
            sumCol += array[j][i]**2
        B.append(sumCol)
        sumCol = 0
    return B
     
print(sum_sq_row(testA))
print(sum_sq_row(testB))
