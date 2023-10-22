import numpy as np
from numba import njit

@njit
def egien(covariance):
    return np.linalg.eig(covariance)