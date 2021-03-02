import numpy as np
import matplotlib.pyplot as plt

def selu(x, alp, l):
    return l*((x > 0)*x + (x <= 0)*(alp * np.exp(x) - alp)) # 람다값에 1 부여

x = np.arange(-5, 5, 0.1)
y = selu(x,2,1)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

# elu 활성화 함수의 변종
# 훈련하는 동안 출력이 평균0과 표준편차 1을 유지하는 경향
