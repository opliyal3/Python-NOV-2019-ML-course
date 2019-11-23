import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
from keras.layers import Flatten

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
print(train_images.shape, test_images.shape)
print(len(train_labels), len(test_labels))
print(np.unique(train_labels))

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
classNames = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(12, 8))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(classNames[train_labels[i]])
# plt.show()

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(layers.Dense(256, activation=tf.nn.relu))
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.Dense(10, activation=tf.nn.softmax))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)