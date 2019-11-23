import tensorflow as tf
import numpy as np
from keras import datasets, utils, layers, Sequential

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print(train_images.shape)
flattenDim = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, flattenDim))
testImages = np.reshape(test_images, (TEST_SIZE, flattenDim))
print(type(trainImages[0]))

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
print(type(trainImages))
trainImages /= 255
testImages /= 255

NUM_DIGITS = 10
trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = utils.to_categorical(test_labels, NUM_DIGITS)
print(trainLabels[0])
print(trainImages[0])

model = Sequential()
model.add(layers.Dense(units=256, activation=tf.nn.relu, input_shape=(flattenDim,)))
model.add(layers.Dense(units=128, activation=tf.nn.relu))
model.add(layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(trainImages, trainLabels, epochs=20)