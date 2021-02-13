# 일반 데이터임 --- 아직 증폭 하기전

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

# ================ ImageDataGenerator정의 ====================
# train ---> 학습필수
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
) # rotation_range 반복값

# test --> 전처리만
test_datagen = ImageDataGenerator(rescale=1/255)

# ===================== generator ===========================
# train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
) # fit과 같은 기능, 실제 이미지파일 전까지 디렉토리 지정
# print(x.shape, y.shape) #(80,150,150,1), (80,)

# test
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

# print(xy_train) # 2 classes(ad, normal)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000020974E58550> -> 딕셔너리형태
# print(xy_train[0]) # -> x, array=4차원(x도 5개)
# [0., 1., 1., 0., 0.] -> y
# print(xy_train[0][0]) # -> x
# print(xy_train[0][0].shape) # -> x (5, 150, 150, 3)왜 5개만 나옴 80개인데? xy_train배치사이즈가 5이므로
# print(xy_train[0][1]) # -> y [0. 1. 0. 1. 1.]
# print(xy_train[0][1].shape) # -> y (5,) ---> xy_train[0] 0은 160장의 데이터 이므로 0~15까지 넣을 수 있다
# print(xy_train[15][1].shape) # -> y (5,) ---> batch_size=10이면 xy_train[0] 0은 160장의 데이터 이므로 0~15까지 넣을 수 있다

np.save('../data/image/brain/npy/keras66_train_x.npy', arr= xy_train[0][0])
np.save('../data/image/brain/npy/keras66_train_y.npy', arr= xy_train[0][1])
np.save('../data/image/brain/npy/keras66_test_x.npy', arr= xy_train[0][0])
np.save('../data/image/brain/npy/keras66_test_y.npy', arr= xy_train[0][1])

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, x_test.shape) # (5, 150, 150, 3) (5, 150, 150, 3)

#실습
#모델 만들자
# ========================= model ==============================
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150,3)))
model.add(Dense(10, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history= model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=150,
    validation_data=xy_test, validation_steps=4
) # xy가 하나로 있는것을 다 한꺼번에 넣어서 할 수 있다

