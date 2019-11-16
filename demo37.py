import tensorflow as tf

vector = [99, 6.9, 3.0, -1.0, 2.4, 5.9, 0.001, 8.5, -0.0000000001]
result1 = tf.nn.relu(vector)
result2 = tf.nn.sigmoid(vector)
print(result1)
print(result2)