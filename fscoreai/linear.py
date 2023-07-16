from tqdm import tqdm
import numpy as np
from fscoreai.utils import sigmoid
from sklearn.preprocessing import normalize

class GeneralizedLinearModel():
    """Generalized Linear Model Base Class

    """
    def __init__(self):
        self.weights_ = None
        self.coef_ = None
        self.intercept_ = None
        self.costs = []
        
    def initialize_weights(self, n_features):
        self.weights_ = np.random.randn(n_features + 1, 1)
        
class LinearRegression(GeneralizedLinearModel):
    """Linear Regression Class
    Uses GeneralizedLinearModel to initialize weights and variables
    """
    def __init__(self):
        super().__init__()
        
    def optimize(self, learning_rate, X, y, tolerance=1e-04):
        y_pred = np.dot(X, self.weights_)
        error = y - y_pred
        cost = (1 / self.n_samples) * np.dot(error.T, error).reshape(-1)
        self.costs.append(cost)
        
        dLdw = - 2 * np.dot(X.T, error) / self.n_samples
        if np.all(np.abs(dLdw) <= tolerance):
            return
        self.weights_ = self.weights_ - (learning_rate * dLdw)
        
    def fit(self, X, y, learning_rate=1e-5, n_epochs=50):
        y = y.reshape(-1, 1)
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.initialize_weights(self.n_features)
        for _ in tqdm(range(n_epochs)):
            self.optimize(learning_rate, X, y)

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
        return np.dot(X, self.weights_)
    
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
                       
        for epoch in tqdm(range(epochs)):
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