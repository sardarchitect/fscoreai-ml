import unittest
import sys
sys.path.append('./')
from primitives.dynamic_array import DynamicArray

class TestDynamicArray(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDynamicArray, self).__init__(*args, **kwargs)
        self.arr = DynamicArray()

    def test_append(self):
        self.arr.append(5)
        self.arr.append(10)
        self.arr.append(20)

        self.assertEqual(self.arr[0], 5, "Append did not work")
        self.assertEqual(self.arr[1], 10, "Append did not work")
        self.assertEqual(self.arr[2], 20, "Append did not work")

    def test_pop(self):
        self.arr.pop()
        self.assertEqual(self.arr[0], 5, "Append did not work")
        self.assertEqual(self.arr[1], 10, "Append did not work")

if __name__ == '__main__':
    unittest.main()