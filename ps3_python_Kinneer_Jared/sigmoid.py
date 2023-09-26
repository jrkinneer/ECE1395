import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    try:
        gz = 1/(1 + math.exp(-z))
    except:
        gz = 0
    return gz

# z = np.linspace(-15, 15, 30)
# gz = np.empty(len(z))
# for i in range(len(z)):
#     gz[i] = sigmoid(z[i])

# plt.plot(z, gz)
# plt.title("z vs gz")
# plt.xlabel("z")
# plt.ylabel("gz")
# plt.show()