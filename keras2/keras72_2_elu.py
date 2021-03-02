# https://subinium.github.io/introduction-to-activation/
# elu : 지수함수를 이용해 입력이 0이하인 경우 부드럽게 깍아준다

import numpy as np
import matplotlib.pyplot as plt

def elu(x, a=1):
    return (x>0)*x + (x<=0)*(a*(np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = elu(x)

print(x)
print(y) 

plt.plot(x, y)
plt.grid()
plt.show()

# f(α,x) = α(e^x−1), x≤0
#        = x, x>0

