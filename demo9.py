import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
print(type(iris), iris.keys())
print(type(iris.data))
print(type(iris.target))
print(type(iris.target_names), iris.target_names)
print(iris.feature_names)
labels = iris.feature_names
data = iris.data
target = iris.target
print(data[:10])
print(target[:10])
print(np.unique(iris.target_names))
print(np.mean(data, axis=0))
# print(np.mean(data, axis=1))
counter = 1
for i in range(4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        counter += 1
        xData = data[:, i]
        yData = data[:, j]
        x_min, x_max = xData.min()-0.5, xData.max()+0.5
        y_min, y_max = yData.min()-0.5, yData.max()+0.5
        plt.clf()
        plt.scatter(xData, yData, c=target, marker='.', cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
