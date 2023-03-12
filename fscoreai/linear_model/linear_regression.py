import numpy as np

class SimpleLinearRegression():
    """Linear Regression through Ordinary Least Squares (OLS).
    Linear Regression fits a linear hyperplane with coefficients
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
        print("OLS Model Initialized")

    def fit(self, X, y, fit_type = 'stat', lr=1e-8, epochs=50):
        """"
        Parameters
        ----------
        X : numpy array
            N x D dimensional input matrix
        y : numpy array
            N dimensional output/labels matrix
        fit_type : str
            'stat' uses statistical, closed-form solution
            'grad' uses gradient descent to optimize parameters for OLS
        lr : float
            sets learning rate when using fit_type == 'grad'
        epochs : int
            sets number of epochs when using fit_type=='grad'
        Returns
        ------
        null
        """
    
        if (fit_type == 'stat'):    #Evaluate closed-form solution
            if (X.shape[1] < 1):    #If X is one-dimensional
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)
                self.coef_ = np.sum((X - X_mean).T*(y-y_mean), axis=1)\
                    /(np.sum((X - X_mean)**2, axis=0)) 
                self.intercept_ = y_mean - (self.coef_ * X_mean)
                return self
            else:   # If X is multi-dimensional
                X = np.hstack((np.ones((X.shape[0],1)), X))
                beta = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T,y))
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
                return self

        if (fit_type == 'grad'): # Evaluate gradient descent solution:
            self.lr = lr  #   Learning rate
            self.epochs = epochs
            self.n, self.d = X.shape

            for epoch in range(self.epochs):
                for j in range(self.d):
                    y_pred = self.predict(X)
                    d_coef = - (1 / self.n) * np.sum((y - y_pred).dot(X[:,j])) #Derivative w.r.t. self.coef_
                    d_intercept = - (1 / self.n) * np.sum(y - y_pred) #Derivative w.r.t. self.intercept_
                    self.coef_ -=  self.lr * d_coef          #    Update self.coef_ 
                    self.intercept_ -=  self.lr * d_intercept  #    Update self.intercept_
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
