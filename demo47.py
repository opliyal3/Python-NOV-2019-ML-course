import numpy
from keras.datasets import imdb
from matplotlib import pyplot

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(numpy.unique(y_train), numpy.unique(y_test))
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X[0])
print(X.shape)
print(y.shape)
print(len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print(result[:50])
print(f'comments mean={numpy.mean(result)}, std={numpy.std(result)}')

pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()
