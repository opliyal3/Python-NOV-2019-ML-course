import numpy as np
from sklearn.svm import SVC


x = np.array([[-1, -1], [-2, -1], [-3, -3], [1, 1], [2, 1], [3, 2]])  # 6個點
y = np.array([1, 1, 1, 2, 2, 2])  # 1/ 2 分類
classifier = SVC()
classifier.fit(x, y)
print(classifier)
print("predict:", classifier.predict([[-2, -3], [2, 3], [4, 4], [0, 0], [0.5, -0.5]]))
