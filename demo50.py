scores = [0.3, 0.4, 0.3]
import numpy as np


def manualSoftmax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(manualSoftmax(scores))

import tensorflow as tf

result2 = tf.nn.softmax(scores)
print(result2)

scores2 = [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
result3 = tf.nn.softmax(scores2)
print(result3)

