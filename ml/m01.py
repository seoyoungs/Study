import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # 0.1 간격으로 0부터 10까지
y = np.sin(x) # 싸인 함수 그래프

plt.plot(x, y)
plt.show()



