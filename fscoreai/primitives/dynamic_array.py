import ctypes

class Dynamic_Array():
    def __init__(self):
        self.length = 0
        self.capacity = 1
        self.array = self.make_array(self.capacity)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            return IndexError("Index out of range")
        return self.array[index]
    
    def append(self, value):
        if self.length == self.capacity:
            self.double_capacity()
        self.array[self.length] = value
        self.length += 1
    
    def insert(self, index, value):
        if index < 0 or index > self.length:
            return IndexError("Index out of range")
        if self.length == self.capacity:
            self.double_capacity()
        for i in range(self.length-1 , index-1, -1):
            self.array[i + 1] = self.array[i]
        self.array[index] = value
        self.length += 1

    def pop(self):
        if self.length == 0:
            print("Array is empty")
            return
        self.array[self.length - 1] = 0
        self.length -= 1

    def delete(self, index):
        if self.length == 0:
            print("Array is empty")
            return
        if index < 0 or index >= self.length:
            return IndexError("Index out of range")
        if index == self.length - 1:
            self.pop()
        for i in range(index, self.length - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.length - 1] = 0
        self.length -= 1
            
    def make_array(self, capacity):
        return (capacity * ctypes.py_object)()
    
    def double_capacity(self):
        new_capacity = self.capacity * 2
        temp_arr = self.make_array(new_capacity)

        for i in range(self.length):
            temp_arr[i] = self.array[i]

        self.array = temp_arr
        self.capacity = new_capacity