# linear and plot

import matplotlib.pyplot as plt
import numpy as np

range1 = [-1, 3] # X :-1 to 3
p = np.array([20])
plt.plot(range1, p * range1 + 5, c = "r")
plt.show()
