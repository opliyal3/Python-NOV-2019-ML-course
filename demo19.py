import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(X.shape)
kmeans = KMeans(n_clusters=4) # 分4類
kmeans.fit(X)
print(kmeans.labels_) # 標籤
print(kmeans.cluster_centers_) # 4個中心點
print(kmeans.inertia_)

colors = ['r', 'g', 'y', 'b']
markers = ['p', '*', '.', 'v'] # https://matplotlib.org/3.1.1/api/markers_api.html

for i in range(4):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i], s=30)
    print(dataX.size)
plt.show()
