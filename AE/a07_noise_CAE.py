# ============== 실습 ==================
# 모델 구성 CNN으로 하기

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 자리수는 맞추는데 명시는 안한다 x만 필요

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_train_out = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape) 
# 원래는 0~1까지 되는데  0.1더해서 0~1.1까지로 된다. 그래서 방지 위해 밑에 clip을 쓴다
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
# 최대값을 (0~1)로 제한--> 노이지가 있는것을 일부로 만듬
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

'''
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,),
                  activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# UpSampling2D 와 Conv2DTranspose의 차이
# http://machinelearningkorea.com/2019/08/25/upsampling2d-%EC%99%80-conv2dtranspose%EC%9D%98-%EC%B0%A8%EC%9D%B4/
'''

#================== CONV2D로 하고 Dense로 바꾸기(모양을 transpose로 잡아준다) ================
def autoencoder(hidden_layer_size):
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_1 = x

    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_2 = x

    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x

    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_1)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)

    return model

'''
# ============ 영리언니의 모델함수로 만들기 ☆ ===========================================
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu'
                     , padding='same', input_shape= (28, 28, 1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(16, 2, 1))
    model.add(UpSampling2D((2, 2))) # 맥스 풀 했으니 업해주기!!!! 모양 유지
    model.add(Flatten())
    model.add(Dense(units=hidden_layer_size/2))
    model.add(Dense(units=hidden_layer_size))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
'''

#hidden_layer_size = 154는 95% PCA Peach를 뜻한다(가장 안정적 복원)
model = autoencoder(hidden_layer_size = 154)
model.compile(optimizer='adam', loss = 'binary_crossentropy',
                 metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=10) #  번갈아 가면서 훈련 시키기 위함

output = model.predict(x_test_noised) #노이즈 제거 됐는지 확인

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
       (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3,5, figsize = (20,7))

# 이미지 다섯개 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

# 원본, 노이즈, 오토인코더 한것
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 노이즈
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('NOISE', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



