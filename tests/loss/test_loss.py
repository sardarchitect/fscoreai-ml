import os
import sys
import pytest
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from math import isclose
from fscoreai.loss.loss import *

@pytest.fixture
def sample_data():
    a = np.array([[ 3.84305282,  5.85540634,  1.34679585, -2.30944547, -0.91812897,
        -1.5900476 , -1.19033388,  0.68946864, -2.29437608,  1.37138265,
         0.99107096,  1.19243385,  1.34343534,  1.3212232 ,  2.2081419 ],
       [-3.64221525, -1.85831677, -0.41928278,  0.90282337,  1.64831909,
        -1.47685588,  4.80267432,  0.12060631,  1.56508166, -0.41184103,
         0.178994  , -0.6938113 ,  5.38175905, -1.13830171,  1.78628003]])
    
    b = np.array([[ 5.08419105,  5.54174599,  0.98403395, -2.1016293 , -1.51853519,
        -0.67200586,  1.1375822 ,  0.38584016, -3.07862204, -0.54330592,
         2.16292175,  1.49862194,  1.06518289,  0.32603625,  2.23604564],
       [-3.55431546, -2.22426342,  0.41081741,  2.20431978,  2.50025398,
        -2.37324654,  3.52377817,  0.54048704,  1.68813355, -0.87675301,
         0.92399089, -3.22339077,  6.84839793, -2.2338433 ,  2.86227653]])
    
    return a, b

def test_should_check_sample_data_shape(sample_data):
    a, b = sample_data[0], sample_data[1]
    assert a.shape == (2, 15)
    assert b.shape == (2, 15)

def test_should_verify_mean_absolute_error_output(sample_data):
    a, b = sample_data[0], sample_data[1]
    assert isclose(mean_absolute_error(a, b), 0.8428982796666666, abs_tol=1e-5)

def test_should_verify_squared_error_output(sample_data):
    a, b = sample_data[0], sample_data[1]
    assert isclose(squared_error(a, b), 32.90479609110383, abs_tol=1e-8)

def test_should_verify_mean_squared_error_output(sample_data):
    a, b = sample_data[0], sample_data[1]
    assert isclose(mean_squared_error(a, b), 1.0968265363701275, abs_tol=1e-8)

def test_should_verify_root_mean_squared_error_output(sample_data):
    a, b = sample_data[0], sample_data[1]
    assert isclose(root_mean_squared_error(a, b), 1.0472948660096293, abs_tol=1e-8)

def test_should_verify_root_mean_squared_log_error_output(sample_data):
    a, b = np.abs(sample_data[0]), np.abs(sample_data[1])
    assert isclose(root_mean_squared_log_error(a, b), 0.10948313065081614, abs_tol=1e-8)

def test_should_verify_r_squared_output(sample_data):
    a, b = sample_data[0].reshape(-1), sample_data[1].reshape(-1)
    assert isclose(r_squared(a, b), 0.7808862363889533, abs_tol=1e-8)