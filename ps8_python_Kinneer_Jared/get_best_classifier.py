import numpy as np
from decision_tree import decision_tree
from knn import knn
from log_reg import log_reg
from random_forest import rf
from svm import svm_

results = np.zeros((5, 6))

results[0] = svm_()
results[1] = knn()
results[2] = log_reg()
results[3] = decision_tree()
results[4] = rf()

print(results)

X_test_results = results[:, 5]
print(np.where(X_test_results == min(X_test_results))[0])
