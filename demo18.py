import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]
]
print(X.shape, X[:10])
for element in X:
    plt.scatter(element[0], element[1], c='black', s=7)

plt.show()
print("min,max", np.min(X), np.max(X))
print("min, max with row,", np.min(X, axis=0), np.max(X, axis=0))
k = 3
C_x = np.random.uniform(np.min(X, axis=0)[0], np.max(X, axis=0)[0], size=k)
C_y = np.random.uniform(np.min(X, axis=0)[1], np.max(X, axis=0)[1], size=k)
# C_x = np.random.randint(np.min(X, axis=0)[0], np.max(X, axis=0)[0], size=k)
# C_y = np.random.randint(np.min(X, axis=0)[1], np.max(X, axis=0)[1], size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
plt.scatter(C_x, C_y, marker='*', s=200, c='#40FFEE')
plt.show()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


pointA = np.array([[1, 1, 1]])
pointB = np.array([[2, 2, 2]])
print(dist(pointA, pointB))

C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if current_cluster[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#40FFEE')
    plt.title('delta will be :%.4f' % delta)
    plt.show()


while delta != 0:
    print("start a new iteration")
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)
