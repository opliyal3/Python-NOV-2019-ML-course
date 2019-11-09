# check version
import sys

import sklearn
import pandas
import numpy
import scipy
import matplotlib

print("its work at ,", sys.executable)
print("numpy version = {}".format(numpy.__version__))
print("scipy version = {}".format(scipy.__version__))
print(f"sklearn version = {sklearn.__version__}")
print(f"pandas version = {pandas.__version__}, matplotlib = {matplotlib.__version__}")
