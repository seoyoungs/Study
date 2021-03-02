import numpy as np
import matplotlib.pyplot as plt

def LeakyReLU(x):
    return np.maximum(0.01*x,x)

x = np.arange(-5, 5, 0.1)
y = LeakyReLU(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

# ReLU와 거의 비슷한 형태를 갖습니다
# 일반적으로 알파를 0.01 -> (0.01*x,x)
