import unittest
import sys
sys.path.append("/Users/fscoreai/repos/github.com/sardarchitect/fscoreai-ml")
from fscoreai.primitives import dynamic_array

class TestDynamicArray(unittest.TestCase):
    def __init__(self):
        self.arr = dynamic_array.Dynamic_Array()

    def test_append(self):
        self.arr.append(5)
        self.assertEqual(self.arr[0], 5, "Append did not work")

if __name__ == '__main__':
    unittest.main()
