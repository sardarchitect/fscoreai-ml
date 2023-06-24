import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from src.primitives.set import Set

x = Set()
x.append(5)
x.display_array()
x.append(10)
x.display_array()
x.pop()
x.display_array()
x.insert(0,2)
x.insert(0,10)
x.insert(0,6)
x.insert(0,8)
x.display_array()

x.delete(0)
x.delete(0)
x.delete(0)
x.delete(0)
x.delete(0)
x.display_array()
