# =========== auotoencoder ==============
# auotoencoder는 y가 없다
# 비지도 학습
#  노드의 모양이 대칭(앞뒤가 똑같은~~~)

# ============ # 하나는 똑같이 만들고 하나 다르게 만들어서 성능비교
# 2개를 만드는데 하나는 랜덤하게 하나는 히든으로 구성
# 2개의 성능비교

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

# ==================== 모델 2가지======================================
# -------------------- 원칙(원형 - 수렴 - 원형 : 앞뒤가 똑같은!!)
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,),
                     activation= 'relu'))
    model.add(Dense(units=hidden_layer_size/2))
    model.add(Dense(units=hidden_layer_size/2))
    model.add(Dense(units=hidden_layer_size/10))
    model.add(Dense(units=hidden_layer_size/3))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# # -------------------- 내맘대로 인코딩
# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape = (784,),
#                      activation= 'relu'))
#     model.add(Dense(units=hidden_layer_size*2))
#     model.add(Dense(units=hidden_layer_size/2))
#     model.add(Dense(units=hidden_layer_size))
#     model.add(Dense(units=hidden_layer_size*4))
#     model.add(Dense(units=hidden_layer_size/5))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model

# ============== 히든 레이어 =============
model_154 = autoencoder(hidden_layer_size= 154) # 중간 레이어를 많이 준다
# 중간(히든)레이어 작게 잡을수록 원본에 없어지는 것이 많다
model_154.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss = 'mse', metrics=['acc'])
model_154.fit(x_train, x_train, epochs=10)
output = model_154.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize = (20,7))

# 이미지 다섯개를 무작위로 고른다(랜덤)
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력)이미지를 맨위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('INPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



