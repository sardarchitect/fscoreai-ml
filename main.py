from fscoreai.primitives import dynamic_array
from fscoreai.primitives import dynamic_set

arr = dynamic_set.Dynamic_Set()

# Append
arr.append(0)
arr.print()

arr.append(0)
arr.print()

# Insert
arr.insert(0, 1)
arr.print()

arr.insert(2, 2)
arr.insert(3, 2)

arr.print()

# Pop
arr.pop()
arr.print()
arr.append(0)
arr.print()

# Delete
arr.delete(0)
arr.print()

arr.pop()
arr.print()
