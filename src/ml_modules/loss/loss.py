import numpy as np

def squared_error(a, b):
    return np.sum(np.square(a - b))

def mean_squared_error(a, b):
    return np.sum(np.square(a - b))/a.shape[0]

def mean_absolute_error(a, b):
    return np.sum(np.absolute(a - b))/a.shape[0]    

def root_mean_squared_error(a, b):
    return np.sqrt(mean_squared_error(a, b))

def root_mean_squared_log_error(a, b):
    return np.log(np.sqrt(mean_squared_error(a, b)))
    
def r_squared(y, y_pred):
    y_mean = y.copy()
    for i in range(y.shape[0]):
        y_mean[i] = np.mean(y)

    SSM = squared_error(y, y_mean)
    SSR = squared_error(y, y_pred)
    return 1 - (SSR / SSM)
