import ctypes

class DynamicArray():
    def __init__(self):
        self.length = 0
        self.capacity = 1
        self.array = self._make_array(self.capacity)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if (index < 0) or (index >= self.length):
            return IndexError("Index out of range")
        return self.array[index]

    def _resize_array(self, new_capacity):
        temp_arr = self._make_array(new_capacity)
        for i in range(self.length):
            temp_arr[i] = self.array[i]
        self.array = temp_arr
        self.capacity = new_capacity

    def _make_array(self, capacity):
        return (capacity * ctypes.py_object)()

    def insert(self, index, value):
        if (index < 0) or (index >= self.length):
            return IndexError("Index out of range")

        if self.length == self.capacity:
            self._resize_array(2 * self.capacity)

        for i in range(self.length-1, index, -1):
            self.array[i + 1] = self.array[i]
        self.array[index] = value
        self.length += 1

    def delete(self):
        pass

    def append(self, value):
        if self.length == self.capacity:
            self._resize_array(2 * self.capacity)

        self.array[self.length] = value
        self.length += 1

    def pop(self):
        self.array[self.length - 1] = 0
        self.length -= 1

    def print(self):
        print(f"{self.length}/{self.capacity}")
        for i in range(self.length):
            print(self.array[i], end=' ')
        print('\n')