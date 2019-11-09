import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)

print(regression1.coef_)
print(regression1.intercept_)
plt.scatter([[0], [1], [2]], [1, 4, 5.5], c='g')
plt.scatter([[1], [3], [8]], [1, 4, 5.5], c='b')
plt.show()
# y=a1x1+a2x2+1.5
print(f'for x1, coef={regression1.coef_[0]}')
print(f'for x2, coef={regression1.coef_[1]}')
# predict
result1 = regression1.predict([[0.8, 0.8], [2, 1], [10, 14]])
print(result1)
# r squared
print(regression1.score([[0, 1], [1, 3], [2, 8]], [1, 4, 5.5]))
print(regression1.score([[0, 1], [1, 3], [2, 8]], [2, 3, 5.5]))
