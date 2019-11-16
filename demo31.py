import tensorflow as tf
import numpy as np

print(tf.__version__)

a = np.array([5, 3, 8])
b = np.array([3, -1, 8])
c = np.add(a, b)
print(c)

# int32
a2 = tf.constant([5, 3, 8])
b2 = tf.constant([3, -1, 2])
c2 = tf.add(a2, b2)
print(c2)
