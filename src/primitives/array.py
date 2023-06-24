import ctypes
class ArrayBase():
    '''
    Base Array class
    '''
    def __init__(self):
        self.length = 0
        self.capacity = 1
        self.array = self._make_array(self.capacity)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if (index < 0) or (index >= self.length):
            raise IndexError("Index out of range")
        return self.array[index]

    def _resize(self, new_capacity):
        temp_arr = self._make_array(new_capacity)
        for i in range(self.length):
            temp_arr[i] = self.array[i]
        self.array = temp_arr
        self.capacity = new_capacity

    def _make_array(self, capacity):
        return (capacity * ctypes.py_object)()

    def display_array(self):
        print(f"{self.length}/{self.capacity}")
        if self.length == 0:
            print("Empty array")
            return
        for i in range(self.length):
            print(self.array[i], end=' ')
        print('\n')

    def search_linear(self, value):
        for i in range(self.length):
            if self.array[i] == value:
                return i
        print("Value not found in array")
        return
    
    def search_binary(self, value, sorted=False):
        if not sorted:
            self.sort_bubble()
        
        lower_bound = 0
        upper_bound = self.length - 1

        while lower_bound <= upper_bound:
            midpoint = (upper_bound + lower_bound) // 2
            value_at_midpoint = self.array[midpoint]
            if value == value_at_midpoint:
                return midpoint
            
            elif value < value_at_midpoint:
                upper_bound = midpoint - 1
            
            elif value > value_at_midpoint:
                lower_bound = midpoint + 1
        

    def sort_bubble(self):
        upper_bound = self.length - 1
        sorted = False

        while not sorted:
            sorted = True
            for i in range(upper_bound):
                if self.array[i] > self.array[i + 1]:
                    self.array[i], self.array[i + 1] = self.array[i + 1], self.array[i]
                    sorted = False
            upper_bound -= 1
            
        return

    def sort_selection(self):
        for i in range(0, self.length - 1):
            lowest_number_index = i
            for j in range(i + 1, self.length):
                if self.array[j] < self.array[lowest_number_index]:
                    lowest_number_index = j
            
            if lowest_number_index != i:
                temp = self.array[i]
                self.array[i] = self.array[lowest_number_index]
                self.array[lowest_number_index] = temp

        return
    
    def sort_insertion(self):
        pass
    
class DynamicArray(ArrayBase):
    '''
    Dynamic Array class
    '''
    def __init__(self):
        super().__init__()
        
    def insert(self, index, value):
        if index < 0 or index > self.length:
            raise IndexError("Index out of range")
        
        if self.length == self.capacity:
            self._resize(2 * self.capacity)

        for i in range(self.length - 1, index -1, -1):
            self.array[i + 1] = self.array[i]
        
        self.array[index] = value
        self.length += 1
        
    def delete(self, index):
        if self.length == 0:
            print ("Array is empty. Deletion not possible.")
            return

        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")
        
        if self.index == self.length - 1:
            self.array[index] = 0
            self.length -= 1
            return

        for i in range(index, self.length - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.length - 1] = 0
        self.length -= 1

    def append(self, value):
        if self.length == self.capacity:
            self._resize(2 * self.capacity)

        self.array[self.length] = value
        self.length += 1

    def pop(self):
        if self.length == 0:
            print ("Array is empty. Deletion not possible.")
            return
        self.array[self.length - 1] = 0
        self.length -= 1

class Set(DynamicArray):
    '''
    Set class
    '''
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
        
    def append(self, value):
        if self.__contains__(value):
            raise ValueError("Item already exists in set")
        super().append(value)

class OrderedArray(DynamicArray):
    '''
    Dyanmic Array Class
    '''
    def __init__(self):
        super().__init__()

    def find_index_for_insert(self, value):
        index = self.length
        for i in range(0, self.length):
            if self.length == 0:
                index = 0
            if self.array[i] > value:
                return i
        
        return index

    def insert(self, value):
        index = self.find_index_for_insert(value)
        super().insert(index, value)

    def delete(self, value):
        index = False
        for i in range(self.length):
            if self.array[i] == value:
                index = i
        if index:
            super().delete(index)

    def sort_bubble(self):
        return super().sort_bubble(sorted=True)