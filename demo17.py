from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [2, 4], [4, 2], [4, 0], [4, 8], [4, 9]])
kmeans = KMeans(n_clusters=2).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[0, 0], [4, 4], [5, 10]]))
print(kmeans.inertia_)
