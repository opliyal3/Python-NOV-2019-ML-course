from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))

X = iris["data"][:, 3:]
y = (iris["target"]==2).astype(np.int)
print(X) # breakpoint here

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_probability = regression1.predict_proba(X_new)
plt.plot(X, y, 'gs')
plt.plot(X_new, y_probability[:, 1], 'r-', label='virginica')
plt.plot(X_new, y_probability[:, 0], 'b--', label='not virginica')
plt.legend(fontsize=14)
plt.show()

print(regression1.predict([[1.3], [1.5], [1.7], [1.9]]))
print(regression1.predict_proba([[1.3], [1.5], [1.7], [1.9], [2.5]]))