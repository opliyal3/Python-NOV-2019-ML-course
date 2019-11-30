import tensorflow as tf
import numpy as np
from keras import datasets, utils, layers, Sequential, callbacks
from keras.layers import Flatten

# load model
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print(train_images.shape)

#convert input image to float
trainImages = train_images.astype(np.float32)
testImages = test_images.astype(np.float32)
print(type(trainImages))
trainImages /= 255
testImages /= 255


model = Sequential()
# whatever data is, flatten
model.add(Flatten(input_shape=(28,28)))
model.add(layers.Dense(units=256, activation=tf.nn.relu))
model.add(layers.Dense(units=128, activation=tf.nn.relu))
# for categorical, (more than 1)
model.add(layers.Dense(units=10, activation=tf.nn.softmax))
# 2. change loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 1. change back to [0-->n-1]
model.fit(trainImages, train_labels, epochs=20)

tbCallback = callbacks.TensorBoard(log_dir='C:\\tensor_log', histogram_freq=0, write_graph=True, write_images=True)

model.fit(trainImages, train_labels, epochs=20, callbacks=[tbCallback])