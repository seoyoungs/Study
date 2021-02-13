# 실습
# cifar10을 flow로 구성해 완성
# ImageDataGenerator/fit_generator--> numpy 저장 완성
# https://whereisend.tistory.com/53

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ===================== 전처리 ==========
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
) 
datagen.fit(x_train)

# ================= model ==============================
def modeling():
    model = Sequential()
    model.add(Conv2D(filter=128, kernel_size=(2,2), padding='same',
                     strides=1, activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(4,4))
    model.add(Conv2D(64,2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # ============ 3. 컴파일, 훈련 ==============================
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs=20)
