import tensorflow as tf

# tensorflow1
tf.compat.v1.disable_eager_execution()


def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


with tf.compat.v1.Session() as session1:
    area = computeArea(tf.compat.v1.constant([[5.0, 3.0, 4.0], [4.2, 5.3, 6.4]]))
    result = session1.run(area)
    print(result)
