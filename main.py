from fscoreai.primitives import dynamic_array

def print_arr(arr):
    for i in range(len(arr)):
        print(arr[i])

arr = dynamic_array.Dynamic_Array()

##
arr.append(1)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.append(2)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.append(3)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)

## 
arr.insert(3, 4)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.insert(4, 5)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.insert(5, 6)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)

## 
arr.insert(0, -1)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.insert(0, -2)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.insert(0, -3)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)

## 
arr.pop()
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.pop()
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.pop()
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)

## 
arr.delete(0)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.delete(0)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)
arr.delete(0)
print(f"Len/Cap: {len(arr)} / {arr.capacity}")
print_arr(arr)