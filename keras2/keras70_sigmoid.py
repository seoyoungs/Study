import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()

# 그림보면 y축에 0~1사이에 수렴!


