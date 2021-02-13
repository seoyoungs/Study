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
'''
rotation_range: 이미지 회전 범위 (degrees)
width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
shear_range: 임의 전단 변환 (shearing transformation) 범위
zoom_range: 임의 확대/축소 범위
horizontal_flip: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

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

# ================== model ============================
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150,3)))
model.add(Dense(10, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history= model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=100,
    validation_data=xy_test, validation_steps=4
) # xy가 하나로 있는것을 다 한꺼번에 넣어서 할 수 있다
'''
steps_per_epoch은 한 번 epoch 돌 때, 데이터를 몇 번 볼 것인가를 정해준다.
validation_steps는 한 번 epoch 돌 고난 후, validation set을 통해 validation accuracy를 측정할 때 validation set을 몇 번 볼 것인지를 정해준다.
'''



# ================== 그림 =======================
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화하자
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('loss&acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

print("val_acc :", np.mean(val_acc))
print('acc: ', acc[-1])


'''
val_acc : 0.4785000023245811
acc:  0.5
'''
