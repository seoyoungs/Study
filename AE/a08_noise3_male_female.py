# keras67_1 남녀에 noise넣고
# 기미 주근깨 제거해라
# 잡음 넣고 안넣은거랑 같이 훈련
# (numpy 만들어논 파일로 generator불러오기)

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

train_x = np.load('C:/data/npy/keras67_train_x.npy')
train_y = np.load('C:/data/npy/keras67_train_y.npy')
test_x = np.load('C:/data/npy/keras67_val_x.npy')
test_y = np.load('C:/data/npy/keras67_val_y.npy')

print(train_x.shape, test_x.shape) # (16, 150, 150, 3) (16, 64, 64, 3) # 3은 컬러, 1은 흑백
# 이미 앞에서 1/255를 해서 안해도 된다

x_train_noised = train_x + np.random.normal(0, 0.1, size = train_x.shape) 
# 원래는 0~1까지 되는데  0.1더해서 0~1.1까지로 된다. 그래서 방지 위해 밑에 clip을 쓴다
x_test_noised = test_x + np.random.normal(0, 0.1, size = test_x.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
# 최대값을 (0~1)로 제한--> 노이지가 있는것을 일부로 만듬
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

'''
#================== 수현오빠 CONV2D로 하고 Dense로 바꾸기(모양을 transpose로 잡아준다) ================
# con2d랑 tranpose는 개수 맞춰서 해주기
def autoencoder():
    inputs = Input(shape=(150,150,3))
    layer1 = Conv2D(64, (3,3), strides=2, padding='same')(inputs)
    layer1_ = BatchNormalization()(layer1)
    layer1 = LeakyReLU()(layer1_)

    layer2 = Conv2D(128, (3,3), strides=2, padding='valid')(layer1)
    layer2_ = BatchNormalization()(layer2)
    layer2 = LeakyReLU()(layer2_)

    layer3 = Conv2D(256, (3,3), strides=2, padding='valid')(layer2)
    layer3_ = BatchNormalization()(layer3)
    layer3 = LeakyReLU()(layer3_)

    layer4 = Conv2D(512, (3,3), strides=2, padding='same')(layer3)
    layer4_ = BatchNormalization()(layer4)
    layer4 = LeakyReLU()(layer4_)

    layer5 = Conv2DTranspose(256, (3,3), strides=2, padding='same')(layer4)
    layer5 = BatchNormalization()(layer5)
    layer5 = layer5 + layer3_
    layer5 = Dropout(0.5)(layer5)
    layer5 = ReLU()(layer5)

    layer6 = Conv2DTranspose(128, (3,3), strides=2, padding='valid')(layer5)
    layer6 = BatchNormalization()(layer6)
    layer6 = layer6+layer2_
    layer6 = Dropout(0.5)(layer6)
    layer6 = ReLU()(layer6)

    layer7 = Conv2DTranspose(64, (3,3), strides=2, padding='valid')(layer6)
    layer7 = BatchNormalization()(layer7)
    layer7 = layer7+layer1_
    layer7 = Dropout(0.5)(layer7)
    layer7 = ReLU()(layer7)

    layer8 = Conv2DTranspose(3, (3,3), strides=2, padding='same')(layer7)

    outputs = layer8

    model = Model(inputs=inputs, outputs=outputs)

    return model
'''
# =========== 모델 autoencoder로 묶기 =====================
def autoencoder():# hidden_layer_size -> 컬러라 (150, 150, 3) 그대로 내보내기 Flatten 안해
    model = Sequential()
    model.add(Conv2D(256, 3, activation= 'relu', padding= 'same', input_shape = (150,150,3)))
    model.add(Conv2D(256, 5, activation= 'relu', padding= 'same'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(64, 5, activation= 'relu', padding= 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid')) # 칼라는 3, 흑백은 1

    return model

#hidden_layer_size = 154는 95% PCA Peach를 뜻한다(가장 안정적 복원)
model = autoencoder() # hidden_layer_size = 154
model.compile(optimizer='adam', loss = 'binary_crossentropy',
                 metrics=['acc'])
model.fit(x_train_noised, train_x, epochs=200) #  번갈아 가면서 훈련 시키기 위함
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
    ax.imshow(test_x[random_images[i]].reshape(150, 150, 3), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel('NOISE', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size= 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



