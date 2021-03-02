# https://mlfromscratch.com/activation-functions-explained/#/ 그래프 모양 참조
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

# 0인것 이후부터 나타남(0이하의 값은 다음 레이어에 전달하지 않습니다. 0이상의 값은 그대로 출력합니다)
### 과제
# elu, selu, reaky relu
# keras72_ 로 만들기
