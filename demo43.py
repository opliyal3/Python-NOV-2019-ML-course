import numpy
from keras.layers import Dense
from keras.models import Sequential
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

print(os.getcwd())

dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList[:5])
print(numpy.unique(resultList))


def create_default_model(optimizer='adam', init='uniform'): #1
    model = Sequential()
    model.add(Dense(14, input_dim=8, kernel_initializer=init, activation='relu')) # 2
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # 3
    print(model.summary())
    return model


# , epochs=200, batch_size=20, ==> remove
model = KerasClassifier(build_fn=create_default_model, verbose=0) # 4
# 5
optimizers = ['rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)

result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"result mean={result.mean()}, result std={result.std()}")