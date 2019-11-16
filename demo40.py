import numpy
from keras.layers import Dense
from keras.models import Sequential
import os
from sklearn.model_selection import train_test_split

print(os.getcwd())

dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.3)





print(inputList[:5])
print(numpy.unique(resultList))

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

result_history = model.fit(feature_train, label_train, epochs=200, batch_size=20,
                           validation_data=(feature_test, label_test))

score = model.evaluate(inputList, resultList)

print(type(model.metrics_names))