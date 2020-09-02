from matplotlib import pyplot as plt
import numpy as np

x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
y = np.sin(x)
plt.plot(x, y, 'r')
plt.show()
