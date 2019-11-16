import numpy
from keras.layers import Dense
from keras.models import Sequential
import os
from sklearn.model_selection import StratifiedKFold

print(os.getcwd())

dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList[:5])
print(numpy.unique(resultList))

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

for train, test in fiveFold.split(inputList, resultList):
    result_history = model.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)

    score = model.evaluate(inputList[test], resultList[test])

    totalScores.append(score[1] * 100)

    print(type(model.metrics_names))
    print(f"got a result with loss:{score[0]}, accuracy:{score[1]}")
print(f"total 5 result:{numpy.mean(totalScores)}, std:{numpy.std(totalScores)}")
