import numpy as np

class CustomLinearRegression():
    """Linear Regression through Ordinary Least Squares (OLS).
    
    Linear Regression fits a linear hyperplane with coefficients 
    coef_ = (coef_1,...,coef_d). It minimizes the cost by minimizing the 
    sum of the residuals between observed inputs and the predicted outputs.

    Parameters
    ----------
    None

    Attributes
    ----------    
    coef_ : int
        returns the coefficient
    intercept : int
        returns the y-intercept
    """
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def fit_stats(self, X, y):
        """"
        Parameters
        ----------
        X : numpy array
            N x D dimensional input matrix
        y : numpy array
            N dimensional output/labels matrix
        
        Returns
        ------
        null
        """
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        self.coef_ = np.sum((X - X_mean).T*(y-y_mean), axis=1)/(np.sum((X - X_mean)**2, axis=0)) 
        # self.intercept_ = y_mean - (self.coef_ * X_mean)


    def predict(self, X):
        """"
        Parameters
        ----------
        X : numpy array
            N x D dimensional input matrix

        Returns
        ------
        list
            N dimensional list of model predictions
        """
        return np.dot(X, self.coef_) + self.intercept_