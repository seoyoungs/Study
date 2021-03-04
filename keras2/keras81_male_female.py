# [실습] VGG16으로 만들어 보라
# imagenerator로 vgg로 만들어라

# 실습
# 남자여자 구별
# ImageDataGenerator로 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16, VGG19, Xception
# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

data_generator = ImageDataGenerator(
    rescale = 1. / 255, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 180,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    validation_split = 0.2) 

train_generator = data_generator.flow_from_directory(
    'C:/data/image/ma_female/classifer', 
    target_size =(64, 64), 
    batch_size = 16,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='training')

validation_generator = data_generator.flow_from_directory( 
    'C:/data/image/ma_female/classifer', 
    target_size =(64, 64), 
    batch_size = 16,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='validation')

X_train, y_train = next(train_generator)
X_test, y_test = next(validation_generator)

#============== ImageDataGenerator정의 ====================
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
test_datagen = ImageDataGenerator(rescale = 1./255)

# ===================== generator ===========================
# train

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# validation
xy_val = train_datagen.flow_from_directory(
    'C:/data/image/ma_female/classifer',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

print(xy_train[0][0].shape) # (16, 64, 64, 3)
print(xy_train[0][1].shape) # (16,)

# ======================== model ===========================
from keras.optimizers import Adam
vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(64, 64, 3))
# print(model.weights)

vgg16.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
# vgg16.summary()
# 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
# print(len(vgg16.weights)) # 26
# print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16) # 3차원 -> layer 26개
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

model.compile(
    optimizer=Adam(lr=0.01),
    loss='binary_crossentropy',
    metrics=['acc']
   )
hist = model.fit_generator(generator=xy_train,
                    validation_data=xy_val,
                    epochs=50)

modelpath = '../data/modelCheckpoint/k81_0303_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
history = model.fit_generator(generator=xy_train,
                    validation_data=xy_val,
                    epochs=50,
                    callbacks=[es, cp, lr])

acc = history.history['acc']
val_acc = history.history['val_acc']
print("val_acc :", np.mean(val_acc))
print('acc: ', acc[-1])

import matplotlib.pyplot as plt
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
# plt.show()

model.save('../data/h5/0303_classifer_2.h5')

# ==================== 남녀 구분 이미지로 나타내기 ==========================
# https://thecleverprogrammer.com/2020/11/25/gender-classification-with-python/
import numpy as np
from keras.preprocessing import image

# predicting images
path = "C:/data/image/ma_female/classifer/female/final_1000.png"
img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0]>0.5:
    print("그는 남자다")
else:
    print( "그녀는 여자다")
plt.imshow(img)
plt.show()

