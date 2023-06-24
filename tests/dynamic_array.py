import os
import sys
import pytest
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from src.primitives.dynamic_array import DynamicArray

@pytest.fixture
def array():
    return DynamicArray()

def test_array_length(array):
    assert len(array) == 0

def test_append(array):
    array.append(10)
    array.append(20)
    assert len(array) == 2
    assert array[0] == 10
    assert array[1] == 20

def test_insert(array):
    array.append(10)
    array.append(30)
    array.insert(1, 20)
    assert len(array) == 3
    assert array[0] == 10
    assert array[1] == 20
    assert array[2] == 30

def test_delete(array):
    array.append(10)
    array.append(20)
    array.append(30)
    array.delete(1)
    assert len(array) == 2
    assert array[0] == 10
    assert array[1] == 30

def test_pop(array):
    array.append(10)
    array.append(20)
    array.pop()
    assert len(array) == 1
    assert array[0] == 10

def test_out_of_range_index(array):
    with pytest.raises(IndexError):
        value = array[0]

    with pytest.raises(IndexError):
        array.insert(2, 25)

def test_capacity_resize(array):
    for i in range(10):
        array.append(i)
    assert len(array) == 10
    assert array[0] == 0
    assert array[9] == 9

    array.append(10)
    assert len(array) == 11
    assert array[10] == 10