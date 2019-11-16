import tensorflow as tf

@tf.function
def add(p, q):
    return tf.math.add(p, q) * 2


print(add([1, 2, 3], [4, 5, 6]))