import time
import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from src.primitives.array import *

x = DynamicArray()
y = DynamicArray()

val = 10000
for i in range(0, 10000):
    x.insert(i, val)
    y.insert(i, val)
    val -= 1

tick = time.time()
x.sort_bubble()
tock = time.time()
print(tock - tick)

tick = time.time()
y.sort_selection()
tock = time.time()
print(tock - tick)
