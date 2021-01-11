
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#print('x_train[0]: ', x_train[0])
#print('y_train[0]: ', y_train[0]) # 5
#print(x_train[0].shape) #(28, 28)

#plt.imshow(x_train[0])
plt.imshow(x_train[0], 'gray') #이렇게 gray를 해야 제대로 됨
plt.show()
#그림에서 특성이 없는는 것은 검은색
# 특성이 제일 밝은 것은 255이다. -> 흰색일수록 특성 있음