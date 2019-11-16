import numpy
from keras.layers import Dense
from keras.models import Sequential
import os

print(os.getcwd())

dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList[:5])
print(numpy.unique(resultList))