# =========== auotoencoder ==============
# auotoencoder는 y가 없다
# 비지도 학습
#  노드의 모양이 대칭(앞뒤가 똑같은~~~)

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 자리수는 맞추는데 명시는 안한다 x만 필요

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape= (784,))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['acc'])
autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, 
                 validation_split=0.2)
# autoencoder은 x,y동일하다

decoded_imgs = autoencoder.predict(x_test)

# ============ 차례대로 10개 가져오기
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1) # 원래 이미지 10개 출력하고
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n) 
    plt.imshow(decoded_imgs[i].reshape(28, 28)) #  decoded된것 10개 출력하겠다
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()



