import numpy as np

class SimpleLinearRegression():
    """Linear Regression through Ordinary Least Squares (OLS).
    Linear Regression fits a linear hyperplane with a single coefficient
    (coef_) and an intercept (intercept_). It minimizes the cost by minimizing
    the sum of the residuals between observed inputs and the predicted outputs.

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

    def fit(self, X, y, fit_type = 'stat', alpha=1e-8, epochs=50):
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
    
        if (fit_type == 'stat'):
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            self.coef_ = np.sum((X - X_mean).T*(y-y_mean), axis=1)/(np.sum((X - X_mean)**2, axis=0)) 
            self.intercept_ = y_mean - (self.coef_ * X_mean)
            return self
        
        if (fit_type == 'grad'):
            self.alpha = alpha  #   Learning rate
            self.epochs = epochs
            self.n, self.d = X.shape

            for epoch in range(self.epochs):
                y_pred = self.predict(X)
                d_coef = - (1 / self.n) * np.sum((y - y_pred).dot(X)) #Derivative w.r.t. self.coef_
                d_intercept = - (1 / self.n) * np.sum(y - y_pred) #Derivative w.r.t. self.intercept_
                self.coef_ -=  self.alpha * d_coef          #    Update self.coef_ 
                self.intercept_ -=  self.alpha * d_intercept  #    Update self.intercept_
            return self

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
