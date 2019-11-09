import numpy as np

from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = [1, 1, 1, 2, 2, 2]

classifier1 = GaussianNB()
classifier1.fit(X, Y)

print(classifier1.predict([[-2, -2], [3, 3], [1, 3], [3, 1]]))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print(classifier2.predict([[-2, -2], [3, 3], [1, 3], [3, 1]]))

classifier2.partial_fit([[1, 4]], [1])  # add or update data and test again
classifier2.partial_fit([[-2, 6]], [1])
print(classifier2.predict([[-2, -2], [3, 3], [1, 3], [3, 1]]))  # data r updated so predict numbers changed
