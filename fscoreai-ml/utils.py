import numpy as np

def normalize(a):
    norm = [((i)-min(a))/(max(a)-min(a)) for i in a]
    return norm

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def gaussian(x, mu, sigma):
    return np.exp(-1*((x - mu)**2 /(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))