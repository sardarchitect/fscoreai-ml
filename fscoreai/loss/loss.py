import numpy as np

def mean_squared_error(a, b):
    return np.sum(np.square(a - b))/a.shape[0]