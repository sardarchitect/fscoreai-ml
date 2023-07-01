import numpy as np
from random import seed
from random import randrange

def normalize(arr):
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    normalized_arr = arr/norm
    return normalized_arr

def sigmoid(x):
    # if x >= 0:
    #     return 1/(1+np.exp(-1*x))
    # else:
    #     return np.exp(x)/(1 + np.exp(x))
    return 1/(1+np.exp(-1*x))

def gaussian(x, mu, sigma):
    return np.exp(-1*((x - mu)**2 /(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))


def train_test_split(X, y, train_split, random_seed=42):
    seed(random_seed)
    train_size = train_split * X.shape[0]

    X_train = np.empty((0, X.shape[1]))
    y_train = np.array([])
    X_test = X.copy()
    y_test = y.copy()

    while X_train.shape[0] < train_size:
        randidx = randrange(X_test.shape[0])
        X_train = np.vstack([X_train, X_test[randidx, :]])
        X_test = np.delete(X_test, randidx, axis=0)
        y_train = np.hstack([y_train, y_test[randidx]])
        y_test = np.delete(y_test, randidx, axis=0)

    return X_train, y_train, X_test, y_test