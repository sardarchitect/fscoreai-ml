import numpy as np

class LinearRegression():
    """
    Linear Regression through Ordinary Least Squares (OLS)

    Linear Regression fits a linear hyperplane with coefficients 
    beta = (beta1,...,beta_p). It minimizes the cost by minimizing 
    the sum of the residuals between observed inputs and the predicted
    outputs.

    Parameters
    __________

    Attributes
    __________    
    """
    def __init__(self):
        print("Linear Regression Model Initialized")

    def fit_stats(self, X, y):
        self.beta = np.zeros_like(X)

        self.beta1 = np.sum((X - np.mean(X))*(y - np.mean(y)))
        self.beta1 /= np.sum((X - np.mean(X))**2)
        self.beta2 = np.mean(y) - (self.beta1 * np.mean(X))

    def predict(self, X):
        return np.dot(np.transpose(self.beta1), X) + self.beta2