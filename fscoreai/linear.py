from tqdm import tqdm
import numpy as np
from fscoreai.utils import sigmoid
from fscoreai import loss
from sklearn.preprocessing import normalize

class GeneralizedLinearModel():
    """Generalized Linear Model Base Class

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

class LinearRegression(GeneralizedLinearModel):
    """Linear Regression Class
    Uses GeneralizedLinearModel to initialize weights and variables
    """
    def __init__(self, regularization=None):
        super().__init__()
        self.regularization = regularization

    def __calculate_cost(self, y, y_pred):
        if self.regularization:
            cost = (1 / self.n_samples) * np.sum(np.square(y_pred - y)) + self.regularization(self.weights_)
        else:
            cost = (1 / self.n_samples) * np.sum(np.square(y_pred - y))
        
        self.costs.append(cost)
        return cost
    
    def __hypothesis(self, X):
        return np.dot(X, self.weights_)
    
    def __initialization(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.random.randn(self.n_features + 1, 1)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X
        
    def __update_weights(self, X, y, y_pred, tolerance=1e-04):
        if self.regularization:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.weights_)
        else:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y))
        if np.all(np.abs(dw) <= tolerance):
            return
        self.weights_ = self.weights_ - (self.lr * dw)

    def fit(self, X, y, lr=1e-5, n_epochs=50, verbose=False):
        self.lr = lr
        y = y.reshape(-1, 1)
        X = self.__initialization(X)
        for epoch in tqdm(range(n_epochs)):
            y_pred = self.__hypothesis(X)
            cost = self.__calculate_cost(y, y_pred)
            self.__update_weights(X, y, y_pred)
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
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights_):
        return self.alpha * np.sum(np.abs(weights_))
    
    def derivation(self, weights_):
        return self.alpha * np.sign(weights_)

class LassoRegression(LinearRegression):
    def __init__(self, alpha):
        self.regularization = LassoPenalty(alpha)
        super().__init__(self.regularization)

class RidgePenalty:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, weights_):
        return self.alpha * np.sum(np.square(weights_))
    
    def derivation(self, weights_):
        return self.alpha * 2 * weights_
    
class RidgeRegression(LinearRegression):
    def __init__(self, alpha):
        self.regularization = RidgePenalty(alpha)
        super().__init__(self.regularization)

class ElasticPenalty:
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
    def __init__(self, alpha, alpha_ratio):
        self.regularization = ElasticPenalty(alpha, alpha_ratio=alpha_ratio)
        super().__init__(self.regularization)
















class LogisticRegression():
    def __init__(self):
        self.weights = None
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X, y, lr=0.01, epochs=2):        
        n, d = X.shape
        X = normalize(X, norm='l2', axis=1)
        print(X.shape)
        X = np.hstack((np.ones(shape=(n, 1)), X))
        self.weights = np.random.randn(d+1)
                       
        for _ in tqdm(range(epochs)):
            y_pred = sigmoid(np.dot(X, self.weights))
            dw = (1/n) * X.T.dot(y - y_pred)
            self.weights -= lr * dw
        
        self.intercept_ = self.weights[:1]
        self.coef_ = self.weights[1:]
    
    def predict(self, X):
        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        y_pred = sigmoid(np.dot(X, self.weights.T))
        return np.where(y_pred > 0.5, 1, 0)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy