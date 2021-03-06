# 실습
# 남자여자 구별
# ImageDataGenerator로 완성 , npy save

# 실습
# 남자여자 구별
# ImageDataGenerator로 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

# ================ ImageDataGenerator정의 ====================
# train ---> 학습필수
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split= 0.3
) 

# test --> 전처리만
test_datagen = ImageDataGenerator(rescale=1./255)

# ===================== generator ===========================
# train

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer/',
    target_size=(150,150),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# validation
xy_val = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer/',
    target_size=(150,150),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

print(xy_train[0][0].shape) # (16, 64, 64, 3)
print(xy_train[0][1].shape) # (16,)
# print(xy_train[0][0].shape)
# print(xy_train)

# ================= save ===========================
np.save('C:/data/npy/keras67_train_x.npy', arr= xy_train[0][0])
np.save('C:/data/npy/keras67_train_y.npy', arr= xy_train[0][1])
np.save('C:/data/npy/keras67_val_x.npy', arr= xy_val[0][0])
np.save('C:/data/npy/keras67_val_y.npy', arr= xy_val[0][1])
'''
np.load('C:/data/npy/keras67_train_x.npy')
np.load('C:/data/npy/keras67_train_y.npy')
np.load('C:/data/npy/keras67_val_x.npy')
np.load('C:/data/npy/keras67_val_y.npy')
'''

#================================= model ===================
from keras.optimizers import Adam
model = Sequential([
    # 1st conv
  Conv2D(96, (3,3),strides=(4,4), padding='same', activation='relu', input_shape=(64, 64, 3)),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"),
  BatchNormalization(),
     # 3rd conv
#   Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
#   BatchNormalization(),
#     # 4th conv
#   Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
#   BatchNormalization(),
    # 5th Conv
  Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  BatchNormalization(),
  MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  Flatten(),
#   # To FC layer 1
#   Dense(4096, activation='relu'),
  Dropout(0.5),
  #To FC layer 2
  Dense(15, activation='relu'),
  Dropout(0.5),
  Dense(1, activation='sigmoid')
  ])
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc']
   )
modelpath = '../data/modelCheckpoint/k67_2_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
history = model.fit_generator(generator=xy_train,
                    validation_data=xy_val,
                    epochs=150,
                    callbacks=[es, cp, lr])
acc = history.history['acc']
val_acc = history.history['val_acc']
print("val_acc :", np.mean(val_acc))
print('acc: ', acc[-1])

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

