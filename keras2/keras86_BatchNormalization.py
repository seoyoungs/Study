# tf114 -> base로 바꾸기
# keras45 참고

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#다중분류이므로 원핫코딩, 전처리 해야한다.
##이걸로 전처리 해서 MinMaxscaler안해도 된다.
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.
#이미지 특성맞춰 숫자 바꾸기 x의 최대가 255이므로 255로 나눈다.
#이렇게 하면 0~1 사이로 된다.
#x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
#x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2 # 정규화

model =Sequential()
model.add(Conv2D(filters=50, kernel_size=(3,3), 
                padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(BatchNormalization()) # 정규화
model.add(activation='relu')

model.add(Conv2D(filters=50, kernel_size=(3,3), 
                kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(activation='relu')
# model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=50, kernel_size=(3,3), 
                kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))
model.add(Dense(5))

model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=5)
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=10, batch_size=16, verbose=1,
                 validation_split=0.2,callbacks=[es])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))