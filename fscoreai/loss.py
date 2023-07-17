import numpy as np

def mean_absolute_error(a, b):
    n = a.reshape(-1).shape[0]
    return np.sum(np.absolute(a - b))/n   

def squared_error(a, b):
    return np.sum(np.square(a - b))

def mean_squared_error(a, b):
    n = a.reshape(-1).shape[0]
    return (np.sum((a - b)**2))/n

def root_mean_squared_error(a, b):
    n = a.reshape(-1).shape[0]
    return np.sqrt((np.sum((a - b)**2))/n)

def root_mean_squared_log_error(a, b):
    n = a.reshape(-1).shape[0]
    if (a < 0).any() or (b < 0).any():
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values."
            )
    a = np.log(1 + a)
    b = np.log(1 + b)
    return np.sum((a - b)**2)/n
    
def r_squared(true, predicted):
    true = np.squeeze(true)
    predicted = np.squeeze(predicted)

    assert true.ndim == 1
    assert predicted.ndim == 1

    n = true.shape[0]
    true_mean = np.sum(true, axis=0)/n
    
    SSR = np.sum((true - predicted)**2)
    SSM = np.sum((true - true_mean)**2)    
    
    return 1 - (SSR / SSM)
