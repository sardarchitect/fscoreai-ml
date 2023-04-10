import numpy as np
import pandas as pd

def simulate_linear_data(start=0, stop=1, n=100, beta_0=1.0, beta_1=2.0, eps_mean=0.0, eps_sigma_sq=0.5, seed=42):
    df = pd.DataFrame({"X":np.linspace(start, stop, num=n)})
    df["y"] = beta_0 + beta_1 * df["X"] + np.random.RandomState(seed).normal(
            eps_mean, eps_sigma_sq, n
            )
    X, y = df['X'].to_numpy().reshape(-1,1), df['y'].to_numpy()
    return X, y


def simulate_regression_data(start=0, stop=1, n=30):
    df = pd.DataFrame({"X":np.linspace(start, stop, num=n)})
    df["y"] = (df['X'] > 0.7).astype("int")
    X, y = df['X'].to_numpy().reshape(-1,1), df['y'].to_numpy()
    return X, y


def kc_house_data():
    path = '../datasets/'
    path = path + 'kc_house_data.csv'
    df = pd.read_csv(path)
    X = df[["sqft_living","yr_built"]].to_numpy()
    y = df["price"].to_numpy()
    return X, y  

def salary_data():
    path = '../datasets/'
    path = path + 'salary_data.csv'
    df = pd.read_csv(path)
    X = df['YearsExperience'].to_numpy()
    y = df["Salary"].to_numpy()
    return X, y

def diabetes_data():
    path = '../datasets/'
    path_X = path + 'diabetes_data.csv'
    path_y = path + 'diabetes_target.csv'
    X = pd.read_csv(path_X, delim_whitespace=True)
    y = pd.read_csv(path_y, header=None)
    return X, y

def iris_data():
    path = '../datasets/'
    path = path + 'IRIS.csv'
    df = pd.read_csv(path)
    X = df.drop(columns='species').to_numpy()
    y = df['species'].to_numpy()
    return X, y