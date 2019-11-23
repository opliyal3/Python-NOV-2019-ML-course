from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

dataFrame = read_csv('data\\iris.data', header=None)
print(type(dataFrame), dataFrame.shape)
print(dataFrame.columns)
print(dataFrame.index)
print(type(dataFrame.values))
print(dataFrame.values)
print(dataFrame.values[0,], type(dataFrame.values[0,][0]), type(dataFrame.values[0,][4]))

# cut data
dataset = dataFrame.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.mean(axis=0))
result = preprocessing.scale(features, axis=0, with_mean=True)
print(result.mean(axis=0))

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(type(encoded_Y), encoded_Y[:10], encoded_Y[50:60], encoded_Y[100:110])
dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y[:10], dummy_y[50:60], dummy_y[100:110])


def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=1)
Kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=Kfold)
print("accuracy: %.4f, std: %.4f"%(results.mean(), results.std()))
