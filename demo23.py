import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# Y = np.array([2, 1, 1, 2, 1, 1])
Y = np.array([2, 2, 1, 2, 2, 1])

x_min = -4  # x bar -4 to 4
x_max = 4
y_min = -4  # y bar -4 to 4
y_max = 4

h = .01  # 格點間隔

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
classifier1 = GaussianNB()
classifier1.fit(X, Y)

Z = classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z)

XB, YB, XR, YR = [], [], [], []

index = 0
for index in range(0, len(Y)):
    if Y[index] == 1:
        XB.append(X[index, 0])
        YB.append(X[index, 1])
    elif Y[index] == 2:
        XR.append(X[index, 0])
        YR.append(X[index, 1])

plt.scatter(XB, YB, color='w', label='White')
plt.scatter(XR, YR, color='r', label='Red')
plt.legend()

plt.show()