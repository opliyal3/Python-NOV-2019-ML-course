import numpy
from keras.layers import Dense
from keras.models import Sequential
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

print(os.getcwd())

dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList[:5])
print(numpy.unique(resultList))


def create_default_model():
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


model = KerasClassifier(build_fn=create_default_model, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)

result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"result mean={result.mean()}, result std={result.std()}")