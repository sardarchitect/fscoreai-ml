from src.primitives.dynamic_array import DynamicArray

class Set(DynamicArray):
    def __init__(self):
        super().__init__()
      
    def __contains__(self, value):
        for i in range(self.length):
            if self.array[i] == value:
                return True
        return False

    def insert(self, index, value):
        if self.__contains__(value):
            raise ValueError("Item already exists in set")
        super().insert(index, value)

    def delete(self, index):
        super().delete(index)
        
    def append(self, value):
        if self.__contains__(value):
            raise ValueError("Item already exists in set")
        super().append(value)

    def pop(self):
        super().pop()