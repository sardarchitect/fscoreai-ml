import numpy as np
import pandas as pd
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def simulate_linear_data(start=-1, stop=1, n=100, beta_0=1.0, beta_1=2.0, eps_mean=0.0, eps_sigma_sq=0.5, seed=42):
    X = np.linspace(start, stop, num=n).reshape(-1, 1)
    y = beta_0 + np.dot(X, beta_1)
    y[:, 0] += np.random.RandomState(seed).normal(eps_mean, eps_sigma_sq, n)
    return X, y

def simulate_multilinear_data(n=100, d=3, seed=42):
    X = np.random.randn(n, d)
    true_coef = np.random.randn(1, d)
    true_intercept = np.random.randn(1)
    noise = np.random.randn(n, 1)

    y = np.dot(X, true_coef.T) + true_intercept + noise
    return X, y

def simulate_regression_data(start=0, stop=1, n=30):
    df = pd.DataFrame({"X":np.linspace(start, stop, num=n)})
    df["y"] = (df['X'] > 0.7).astype("int")
    X, y = df['X'].to_numpy().reshape(-1,1), df['y'].to_numpy()
    return X, y

def heart_data():
    path = os.path.join(dir_path, 'heart_data.csv')
    df = pd.read_csv(path)
    X = df[['biking', 'smoking']].to_numpy()
    y = df['heart_disease'].to_numpy().reshape(-1, 1)
    return X, y

def kc_house_data():
    path = os.path.join(dir_path, 'kc_house_data.csv')
    df = pd.read_csv(path)
    X = df[["sqft_living","yr_built"]].to_numpy()
    y = df["price"].to_numpy().reshape(-1, 1)
    return X, y

def salary_data():
    path = os.path.join(dir_path, 'salary_data.csv')
    df = pd.read_csv(path)
    X = df['YearsExperience'].to_numpy().reshape(-1, 1)
    y = df["Salary"].to_numpy().reshape(-1, 1)
    return X, y

def diabetes_data():
    path_X = os.path.join(dir_path, 'diabetes_data.csv')
    path_y = os.path.join(dir_path, 'diabetes_target.csv')
    X = pd.read_csv(path_X, delim_whitespace=True).to_numpy()
    y = pd.read_csv(path_y, header=None).to_numpy()
    return X, y

def iris_data():
    path = '../datasets/'
    path = path + 'IRIS.csv'
    df = pd.read_csv(path)
    X = df.drop(columns='species').to_numpy()
    y = df['species'].to_numpy()
    return X, y