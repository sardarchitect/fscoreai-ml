import numpy as np

def squared_error(a, b):
    return np.sum(np.square(a - b))

def mean_squared_error(a, b):
    n = a.reshape(-1).shape[0]
    return (np.sum((a - b)**2))/n

def mean_absolute_error(a, b):
    n = a.reshape(-1).shape[0]
    return np.sum(np.absolute(a - b))/n   

def root_mean_squared_error(a, b):
    return np.sqrt(mean_squared_error(a, b))

def root_mean_squared_log_error(a, b):
    if (a < 0).any() or (b < 0).any():
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values."
            )
    return np.log(np.sqrt(mean_squared_error(a, b)))
    
def r_squared(y, y_pred):
    y_mean = y.copy()
    for i in range(y.shape[0]):
        y_mean[i] = np.mean(y)

    SSM = squared_error(y, y_mean)
    SSR = squared_error(y, y_pred)
    return 1 - (SSR / SSM)
