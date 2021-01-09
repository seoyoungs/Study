### 인공지능계의 hello wkrld라고 불리는 mnist

'''
- MNIST 데이터셋
: 0부터 9까지의 숫자 이미지로 구성되며, 
60,000개의 트레이닝 데이터와 10,000개의 테스트 데이터로 이루어집니다.
(다중 분류)
'''

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
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





