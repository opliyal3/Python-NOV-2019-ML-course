import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.compat.v1.add(a, b)

with tf.compat.v1.Session() as session:
    result = session.run(c, feed_dict={a: [1, 3, 5], b: [2, 4, 6]})
    print(result)
