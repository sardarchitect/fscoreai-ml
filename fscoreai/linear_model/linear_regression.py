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

    def fit(self, X, y, fit_type = 'stat'):
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
            self.alpha = 1e-11  #   Learning rate
            self.epochs = 50
            self.n, self.d = X.shape

            for epoch in range(self.epochs):
                print("Epoch:", epoch, "Coef:", self.coef_, "Intercept:", self.intercept_)
                y_pred = self.predict(X)
                d_coef = - 2 / self.n * np.sum(np.dot((y - y_pred),X))        #    Derivative w.r.t. self.coef_
                d_intercept = - 2 / self.n * np.sum(y - y_pred)         #    Derivative w.r.t. self.intercept_
                self.coef_ -=  self.alpha * d_coef          #    Update self.coef_ 
                self.intercept_ -=  self.alpha * d_intercept  #    Update self.intercept_

                print("\n d_coef", d_coef, "d_intercept", d_intercept)
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

class MultipleLinearRegression():
    # Implementation reference: https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)
