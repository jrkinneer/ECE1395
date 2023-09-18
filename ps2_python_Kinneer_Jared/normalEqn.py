import numpy as np
def normalEqn(x_train, y_train):
    
    x_transposed = np.transpose(x_train)
    inversed_x_dotproduct = np.linalg.inv(np.dot(x_transposed, x_train))
    x_transposed_y = np.dot(x_transposed, y_train)
    
    final = np.dot(inversed_x_dotproduct, x_transposed_y)
    theta = final
    return theta

#x_train = [x0, x1]
x_train = np.array([[1,0],[1,2],[1,3],[1,4]])
y_train = np.array([4,8,10,12])

theta = normalEqn(x_train, y_train)
print("theta = [",theta[0],", ", theta[1],"]\n")
