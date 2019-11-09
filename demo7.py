import numpy as np

a = np.zeros((10, 2))
print(type(a), a.shape)
print(a)
b = a.T  # transpose
print(b.shape)
print(b)
print(hex(id(a)), hex(id(b)))
c = b.view
print(c)
print(hex(id(c)))

# c+=1
# print(c)
# print(b)
# print(a)

d = np.reshape(b, (5, 4))
print(d)
e = np.reshape(b, (20,))
print(e)
f = np.reshape(b, (20, -1))
print(f)
g = np.reshape(b, (-1, 20))
print(g)
