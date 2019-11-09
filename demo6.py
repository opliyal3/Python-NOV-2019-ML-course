from sklearn import datasets
import matplotlib.pyplot as plt

data1 = datasets.make_regression(10, 6, noise=5)
print(type(data1))
print(data1[0].shape, data1[1].shape)

rX = data1[0]
print(rX)
for pos in range(0, 6):
    r = sorted(rX, key=lambda row: row[pos])
    print(r)

for pos in range(0, 6):
    x1 = rX[:, pos]
    y = data1[1]
    plt.scatter(x1, y)
    plt.show()