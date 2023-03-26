import numpy as np
import pandas as pd

def get_data(name='kc_house_data'):
    path = '../datasets/'
    if name=='kc_house_data':
        path = path + 'kc_house_data.csv'
        df = pd.read_csv(path)
        X = df[["sqft_living","yr_built"]].to_numpy()
        y = df["price"].to_numpy()

    if name=='diabetes':
        path = path + 'diabetes.csv'
        df = pd.read_csv(path)
        X = df["Age (years)"].to_numpy().reshape(-1,1)
        y = df["Class variable"].to_numpy()
    
    if name=='simulate_linear_data':
        data = simulate_linear_data()
        X, y = data['x'].to_numpy().reshape(-1,1), data['y'].to_numpy()
    return X, y  

def simulate_linear_data(start=0, stop=1, n=100, beta_0=1.0, beta_1=2.0, eps_mean=0.0, eps_sigma_sq=0.5, seed=42):
    """
    Simulate a random dataset using a noisy
    linear process.

    Parameters
    ----------
    N: `int`
        Number of data points to simulate
    beta_0: `float`
        Intercept
    beta_1: `float`
        Slope of univariate predictor, X

    Returns
    -------
    df: `pd.DataFrame`
        A DataFrame containing the x and y values.
    """
    df = pd.DataFrame({"x":np.linspace(start, stop, num=n)})
    df["y"] = beta_0 + beta_1 * df["x"] + np.random.RandomState(seed).normal(
            eps_mean, eps_sigma_sq, n
            )
    return df
