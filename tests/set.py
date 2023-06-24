import os
import sys
import pytest
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from src.primitives.set import Set

@pytest.fixture
def set():
    return Set()

def test_set_contains(set):
    print(1 not in set)
    assert 1 not in set

def test_append(set):
    set.append(1)
    set.append(2)
    assert 1 in set
