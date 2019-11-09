import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
neighbors = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
distances, indexes = neighbors.kneighbors(X, return_distance=True)
print(type(distances))
print(distances)
print(type(indexes))
print(indexes)
