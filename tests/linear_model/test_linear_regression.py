import os
import sys
import pytest

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from fscoreai.linear_model.linear_regression import LinearRegression
from sample_data.sample_data import simulate_linear_data, simulate_multilinear_data

@pytest.fixture
def sample_uni_data():
    X, y = simulate_linear_data(n=100, seed=42)
    return (X, y)

@pytest.fixture
def sample_multi_data():
    X, y = simulate_multilinear_data()
    return (X, y)

def test_should_check_sample_data_shape(sample_uni_data, sample_multi_data):
    X_1, y_1 = sample_uni_data[0], sample_uni_data[1]
    X_2, y_2 = sample_multi_data[0], sample_multi_data[1]

    assert X_1.shape == (100, 1)
    assert y_1.shape == (100, 1)
    assert X_2.shape == (100, 3)
    assert y_2.shape == (100, 1)

def test_should_create_linear_regression_class():
    assert LinearRegression() is not None

def test_should_initialize_weights():
    model = LinearRegression()
    assert model.coef_ == None
    assert model.intercept_ == None

def test_should_check_weights_shape_stat_uni(sample_uni_data):
    model = LinearRegression()
    X, y = sample_uni_data[0], sample_uni_data[1]
    
    model.fit_statistical(X, y)
    
    assert model.coef_.shape == (1, 1)
    assert model.intercept_.shape == (1, )

def test_should_check_weights_shape_stat_multi(sample_multi_data):
    model = LinearRegression()
    X, y = sample_multi_data[0], sample_multi_data[1]
    
    model.fit_statistical(X, y)
    
    assert model.coef_.shape == (3, 1)
    assert model.intercept_.shape == (1, )

def test_should_check_weights_shape_grad_uni(sample_uni_data):
    model = LinearRegression()
    X, y = sample_uni_data[0], sample_uni_data[1]

    model.fit(X, y, lr=1e-8, epochs=50)

    assert model.coef_.shape == (1, 1)
    assert model.intercept_.shape == (1, )

def test_should_check_weights_shape_grad_multi(sample_multi_data):
    model = LinearRegression()
    X, y = sample_multi_data[0], sample_multi_data[1]

    model.fit(X, y, lr=1e-8, epochs=50)

    assert model.coef_.shape == (3, 1)
    assert model.intercept_.shape == (1, )

def test_should_check_predictions_shape_uni(sample_uni_data):
    model = LinearRegression()
    X, y = sample_uni_data[0], sample_uni_data[1]
    
    model.fit_statistical(X, y)

    assert model.predict(X).shape == (100, 1)

def test_should_check_predictions_shape_multi(sample_multi_data):
    model = LinearRegression()
    X, y = sample_multi_data[0], sample_multi_data[1]
    
    model.fit_statistical(X, y)

    assert model.predict(X).shape == (100, 1)