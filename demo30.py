import tensorflow as tf

tf.compat.v1.disable_eager_execution()
hello1 = tf.constant('hello tensorflow from python3.7')
print(tf.__version__)
with tf.compat.v1.Session() as session:
    print(session.run(hello1))