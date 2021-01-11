from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

#print('x_train[0]: ', x_train[0])
#print('y_train[0]: ', y_train[0]) # 5
#print(x_train[0].shape) #(28, 28)

#plt.imshow(x_train[0])
plt.imshow(x_train[0], 'gray') #이렇게 gray를 해야 제대로 됨
plt.show()





