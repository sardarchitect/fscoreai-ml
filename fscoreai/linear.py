from tqdm import tqdm
import numpy as np
from fscoreai import utils
from fscoreai import loss
from sklearn.preprocessing import normalize

class GeneralizedLinearModel():
    """Generalized Linear Model Base Class

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

class LinearRegression(GeneralizedLinearModel):
    """Linear Regression Class

    Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.
    Attributes:
        regularization (object): The regularization object used for regularization.
    Args:
        regularization (object): The regularization object to use for regularization.
    """
    def __init__(self, regularization=None):
        super().__init__()
        self.regularization = regularization

    def __calculate_cost(self, y, y_pred):
        """Calculate the cost function for linear regression.

        Args:
            y (ndarray): The true labels.
            y_pred (ndarray): The predicted labels.

        Returns:
            float: The calculated cost.
        """
        if self.regularization:
            cost = (1 / self.n_samples) * np.sum(np.square(y_pred - y)) + self.regularization(self.weights_)
        else:
            cost = (1 / self.n_samples) * np.sum(np.square(y_pred - y))
        
        self.costs.append(cost)
        return cost
    
    def __hypothesis(self, X):
        """Compute the hypothesis function.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The predicted labels.
        """
        return np.dot(X, self.weights_)
        
    def __update_weights(self, X, y, y_pred, tolerance=1e-04):
        """Update the model weights during training.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.
            y_pred (ndarray): The predicted labels.
            tolerance (float, optional): The tolerance threshold for convergence. Defaults to 1e-04.
        """
        if self.regularization:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.weights_)
        else:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y))
        if np.all(np.abs(dw) <= tolerance):
            return
        self.weights_ = self.weights_ - (self.lr * dw)
    
    def __initialization(self, X):
        """Initialize the model weights and add bias term to input features.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The modified input features with bias term.
        """
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.random.randn(self.n_features + 1, 1)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X

    def fit(self, X, y, lr=1e-5, n_epochs=50, verbose=False):
        """Fit the linear regression model to the training data.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.
            lr (float, optional): The learning rate. Defaults to 1e-5.
            n_epochs (int, optional): The number of training epochs. Defaults to 50.
            verbose (bool, optional): Whether to print the training progress. Defaults to False.
        """
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
        """Fit the linear regression model using closed-form solution.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.
        """
        n, d = X.shape
        X = np.hstack((np.ones((n, 1)), X))
        self.weights_ = np.linalg.inv(X.T@X) @ (X.T@y)
        
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
        return

    def predict(self, X):
        """Predict labels for new input features.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The predicted labels.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = self.__hypothesis(X)
        return y_pred
    
    def score(self, X, y):
        """Compute the R-squared score of the model.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.

        Returns:
            float: The R-squared score.
        """
        return loss.r_squared(y, self.predict(X))
        
class LassoPenalty:
    """Lasso (L1) Regularization Penalty.

    Args:
        alpha (float): The regularization parameter.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights_):
        """Compute the Lasso regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            float: The regularization penalty.
        """
        return self.alpha * np.sum(np.abs(weights_))
    
    def derivation(self, weights_):
        """Compute the derivative of the Lasso regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            ndarray: The derivative of the regularization penalty.
        """
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
        """Compute the Ridge regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            float: The regularization penalty.
        """
        return self.alpha * np.sum(np.square(weights_))
    
    def derivation(self, weights_):
        """Compute the derivative of the Ridge regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            ndarray: The derivative of the regularization penalty.
        """
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
        """Compute the Elastic Net regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            float: The regularization penalty.
        """
        l1_contribution = self.alpha_ratio * self.alpha * np.sum(np.abs(weights_))
        l2_contribution = (1 - self.alpha_ratio) * self.alpha * 0.5 * np.sum(np.square(weights_))
        return l1_contribution + l2_contribution
    
    def derivation(self, weights_):
        """Compute the derivative of the Elastic Net regularization penalty.

        Args:
            weights_ (ndarray): The model weights.

        Returns:
            ndarray: The derivative of the regularization penalty.
        """
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


class LogisticRegression(GeneralizedLinearModel):
    """Logistic Regression Class

    Logistic regression is a linear model used for binary classification.

    Attributes:
        regularization (object): The regularization object used for regularization.

    Args:
        regularization (object): The regularization object to use for regularization.
    """
    def __init__(self, regularization=None):
        super().__init__()
        self.regularization = regularization

    def __sigmoid(self, z):
        """Compute the sigmoid function.

        Args:
            z (ndarray): The input to the sigmoid function.

        Returns:
            ndarray: The sigmoid of the input.
        """
        return utils.sigmoid(z)

    def __calculate_cost(self, y, y_pred):
        """Calculate the cost function for logistic regression.

        Args:
            y (ndarray): The true labels.
            y_pred (ndarray): The predicted probabilities.

        Returns:
            float: The calculated cost.
        """
        if self.regularization:
            cost = (-1 / self.n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) + self.regularization(self.weights_)
        else:
            cost = (-1 / self.n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        self.costs.append(cost)
        return cost
    
    def __hypothesis(self, X):
        """Compute the hypothesis function.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The predicted probabilities.
        """
        return self.__sigmoid(np.dot(X, self.weights_))
            
    def __update_weights(self, X, y, y_pred, tolerance=1e-04):
        """Update the model weights during training.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.
            y_pred (ndarray): The predicted labels.
            tolerance (float, optional): The tolerance threshold for convergence. Defaults to 1e-04.
        """
        if self.regularization:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.weights_)
        else:
            dw = (1 / self.n_samples) * np.dot(X.T, (y_pred - y))
        if np.all(np.abs(dw) <= tolerance):
            return
        self.weights_ = self.weights_ - (self.lr * dw)

    def __initialization(self, X):
        """Initialize the model weights and add bias term to input features.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The modified input features with bias term.
        """
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.random.randn(self.n_features + 1, 1)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X
    
    def fit(self, X, y, lr=1e-5, n_epochs=50, verbose=False):
        """Fit the linear regression model to the training data.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.
            lr (float, optional): The learning rate. Defaults to 1e-5.
            n_epochs (int, optional): The number of training epochs. Defaults to 50.
            verbose (bool, optional): Whether to print the training progress. Defaults to False.
        """
        self.lr = lr
        y = y.reshape(-1, 1)
        X = self.__initialization(X)
        for epoch in tqdm(range(n_epochs)):
            y_pred = self.__hypothesis(X)
            # cost = self.__calculate_cost(y, y_pred)
            self.__update_weights(X, y, y_pred)
            if epoch % 100 == 0:
                if verbose:
                    print(f"{epoch}/{n_epochs} | Cost: {cost}")
        
        self.intercept_ = self.weights_[0]
        self.coef_ = self.weights_[1:]
        return

    def predict(self, X, threshold=0.5):
        """Predict labels for new input features.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The predicted labels.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred_proba = self.__hypothesis(X)
        y_pred = np.where(y_pred_proba >= threshold, 1, 0)
        return y_pred
    
    def predict_proba(self, X):
        """Predict class probabilities for new input features.

        Args:
            X (ndarray): The input features.

        Returns:
            ndarray: The predicted class probabilities.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred_proba = self.__hypothesis(X)
        return y_pred_proba
    
    def score(self, X, y):
        """Compute the R-squared score of the model.

        Args:
            X (ndarray): The input features.
            y (ndarray): The true labels.

        Returns:
            float: The R-squared score.
        """
        y_pred = self.predict(X)
        return loss.accuracy_score(y, y_pred)