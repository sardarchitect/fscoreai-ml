import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from src.primitives.array import *

x = OrderedArray()
x.display_array()