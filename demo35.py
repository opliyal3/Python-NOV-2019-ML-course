import tensorflow as tf

# tensorflow2
@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


print("areaSquare: \n")
print(computeArea(tf.constant([[5.0, 3.0, 4.0], [4.2, 5.2, 6.2]])))
