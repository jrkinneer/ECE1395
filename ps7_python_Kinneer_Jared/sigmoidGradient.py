import numpy as np
from predict import sigmoid

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))
    