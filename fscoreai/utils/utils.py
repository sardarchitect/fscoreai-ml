import numpy as np

def normalize(a):
    norm = [((i)-min(a))/(max(a)-min(a)) for i in a]
    return norm

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def gaussian(x, mu, sigma):
    return np.exp(-1*((x - mu)**2 /(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))

import pandas as pd
def get_data(name='kc_house_data'):
    path = '../datasets/'
    if name=='kc_house_data':
        path = path + 'kc_house_data.csv'
        df = pd.read_csv(path)
        X = df["sqft_living"].to_numpy().reshape(-1,1)
        y = df["price"].to_numpy()

    if name=='diabetes':
        path = path + 'diabetes.csv'
        df = pd.read_csv(path)
        X = df["Age (years)"].to_numpy().reshape(-1,1)
        y = df["Class variable"].to_numpy()

    return X, y   
