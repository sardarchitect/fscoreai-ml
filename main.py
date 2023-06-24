from primitives import dynamic_array

arr = dynamic_array.DynamicArray()

# Append
arr.append(1)
arr.append(2)
arr.append(3)
arr.print()

arr.pop()
arr.print()

arr.insert(0, 1)
arr.insert(0, 2)
arr.insert(0, 3)
arr.print()