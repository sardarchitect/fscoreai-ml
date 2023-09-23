from tqdm import tqdm
import numpy as np
from fscoreai import utils
from fscoreai import loss

class GeneralizedLinearModel():
    """
    Generalized Linear Model Base Class

    This is the base class for all generalized linear models. It contains common attributes and methods shared by all models.

    Attributes:
        n_samples (int): The number of samples in the dataset.
        n_features (int): The number of features in the dataset.
        weights_ (ndarray): The learned weights of the model.
        coef_ (ndarray): The coefficients of the model.
        intercept_ (ndarray): The intercept of the model.
        lr (float): The learning rate used during training.
        regularization (object): The regularization object used for regularization.
        costs (list): List of costs during training.
    """
    def __init__(self):
        self.n_samples = None
        self.n_features = None
        self.weights_ = None
        self.coef_ = None
        self.intercept_ = None
        self.lr = None
        self.regularization = None
        self.costs = []
        self.scores = []

class LinearRegression(GeneralizedLinearModel):
    """
    Linear Regression Class

    Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.
    
    Args:
        regularization (object): The regularization object to use for regularization.
    """
    def __init__(self, regularization=None):
        super().__init__()
        self.regularization = regularization

    def __calculate_cost(self, y, y_pred):
        cost = (1 / self.n_samples) * np.sum(np.square(y_pred - y))
        if self.regularization:
            cost += self.regularization(self.weights_)
        
        self.costs.append(cost)
        return cost
    
    def __hypothesis(self, X):
        return np.dot(X, self.weights_)
        
    def __update_weights(self, X, y, y_pred, tolerance=1e-04):
        dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y))
        if self.regularization:
             dw += self.regularization.derivation(self.weights_)
        
        if np.all(np.abs(dw) <= tolerance):
            return
        self.weights_ = self.weights_ - (self.lr * dw)
    
    def __initialization(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.random.randn(self.n_features + 1, 1)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X

    def fit(self, X, y, lr=1e-5, n_epochs=50, verbose=False):
        self.lr = lr
        y = y.reshape(-1, 1)
        X = self.__initialization(X)
        for epoch in tqdm(range(n_epochs)):
            y_pred = self.__hypothesis(X)
            self.__update_weights(X, y, y_pred)
            
            cost = self.__calculate_cost(y, y_pred)
            self.scores.append(self.score(X, y))
            if epoch % 100 == 0:
                if verbose:
                    print(f"{epoch}/{n_epochs} | Cost: {cost}")
        
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
        return
            
    def fit_closed_form(self, X, y):
        n, d = X.shape
        X = np.hstack((np.ones((n, 1)), X))
        self.weights_ = np.linalg.inv(X.T@X) @ (X.T@y)
        
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
        return

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = self.__hypothesis(X)
        return y_pred
    
    def score(self, X, y):
        return loss.r_squared(y, self.predict(X))
        
class LassoPenalty:
    """Lasso (L1) Regularization Penalty.

    Args:
        alpha (float): The regularization parameter.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights_):
        return self.alpha * np.sum(np.abs(weights_))
    
    def derivation(self, weights_):
        return self.alpha * np.sign(weights_)

class LassoRegression(LinearRegression):
    """Lasso Regression Class.

    Lasso regression is a linear regression model with L1 regularization.

    Args:
        alpha (float): The regularization parameter.
    """
    def __init__(self, alpha):
        self.regularization = LassoPenalty(alpha)
        super().__init__(self.regularization)

class RidgePenalty:
    """Ridge (L2) Regularization Penalty.

    Args:
        alpha (float): The regularization parameter.
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, weights_):
        return self.alpha * np.sum(np.square(weights_))
    
    def derivation(self, weights_):
        return self.alpha * 2 * weights_
    
class RidgeRegression(LinearRegression):
    """Ridge Regression Class.

    Ridge regression is a linear regression model with L2 regularization.

    Args:
        alpha (float): The regularization parameter.
    """
    def __init__(self, alpha):
        self.regularization = RidgePenalty(alpha)
        super().__init__(self.regularization)

class ElasticPenalty:
    """Elastic Net Regularization Penalty.

    Args:
        alpha (float): The overall regularization parameter.
        alpha_ratio (float): The ratio of L1 regularization in the penalty.
    """
    def __init__(self, alpha=0.1, alpha_ratio = 0.5):
        self.alpha = alpha
        self.alpha_ratio = alpha_ratio
    
    def __call__(self, weights_):
        l1_contribution = self.alpha_ratio * self.alpha * np.sum(np.abs(weights_))
        l2_contribution = (1 - self.alpha_ratio) * self.alpha * 0.5 * np.sum(np.square(weights_))
        return l1_contribution + l2_contribution
    
    def derivation(self, weights_):
        l1_derivation =  self.alpha_ratio * self.alpha * np.sign(weights_)
        l2_derivation =  (1 - self.alpha_ratio) * self.alpha * (weights_)
        return l1_derivation + l2_derivation
    
class ElasticNetRegression(LinearRegression):
    """Elastic Net Regression Class.

    Elastic Net regression is a linear regression model with a combination of L1 and L2 regularization.

    Args:
        alpha (float): The overall regularization parameter.
        alpha_ratio (float): The ratio of L1 regularization in the penalty.
    """
    def __init__(self, alpha, alpha_ratio):
        self.regularization = ElasticPenalty(alpha, alpha_ratio=alpha_ratio)
        super().__init__(self.regularization)


class LogisticRegression(LinearRegression):
    """Logistic Regression Class

    Logistic regression is a linear model used for binary classification.

    Attributes:
        regularization (object): The regularization object used for regularization.

    Args:
        regularization (object): The regularization object to use for regularization.
    """
    def __init__(self, regularization=None, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.regularization = regularization

    def __calculate_cost(self, y, y_pred):
        """
        Calculate cost using conditional MSE (cross-entropy loss)
        """
        cost = (-1 / self.n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        if self.regularization:
            cost += self.regularization(self.weights_)

        self.costs.append(cost)
        return cost
    
    def __hypothesis(self, X):
        return utils.sigmoid(np.dot(X, self.weights_))
            
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred_proba = self.__hypothesis(X)
        y_pred = np.where(y_pred_proba >= self.threshold, 1, 0)
        return y_pred
    
    def predict_proba(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred_proba = self.__hypothesis(X)
        return y_pred_proba

    def score(self, X, y):
        y_pred_proba = self.__hypothesis(X)
        y_pred = np.where(y_pred_proba >= self.threshold, 1, 0)
        return loss.accuracy_score(y, y_pred)