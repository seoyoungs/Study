####mnist는 시계열 자료가 될 수 있다.
###LSTM으로도 해보기 
# #input_shape=(28*28,1)
# #input_shape=(28*14,2)
# #input_shape=(28*7, 4)
# #input_shape=(7*7,16)

#####주말 과제
####dense모델로 구성 input_shape=(28*28,)
####노트 확인

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#print('x_train[0]: ', x_train[0])
#print('y_train[0]: ', y_train[0]) # 5
#print(x_train[0].shape) #(28, 28)

#plt.imshow(x_train[0])
#plt.imshow(x_train[0], 'gray') #이렇게 gray를 해야 제대로 됨
#plt.show()
#그림에서 특성이 없는는 것은 검은색
# 특성이 제일 밝은 것은 255이다. -> 흰색일수록 특성 있음

#다중분류이므로 원핫코딩, 전처리 해야한다.
##이걸로 전처리 해서 MinMaxscaler안해도 된다.
#x_train= x_train.reshape(60000, 28,28)/255.
#x_test= x_test.reshape(10000, 28,28)/255.
x_train = x_train.reshape((x_train.shape[0], -1,16))/255.
x_test = x_test.reshape((x_test.shape[0], -1,16))/255.
#이미지 특성맞춰 숫자 바꾸기 x의 최대가 255이므로 255로 나눈다.
#이렇게 하면 0~1 사이로 된다.
#(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1) 이것도 됨

# 학습 과정을 단축시키기 위해 학습 데이터의 1/3만 활용한다
from sklearn.model_selection import train_test_split
x_train, _ , y_train, _ = train_test_split(x_train, y_train, 
                          test_size = 0.7, random_state = 7)

#다중분류 y원핫코딩

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model=Sequential()
model.add(LSTM(units=15, activation='relu', input_shape=(7*7,16)))
model.add(Dense(8,activation='relu'))
model.add(Dense(13,activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=15, 
           validation_split=0.2, batch_size=32,verbose=1)

#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=32)
print(loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

'''
Mnist
결과
[0.11182762682437897, 0.9659000039100647]

Dnn
[0.19730423390865326, 0.9408000111579895]

LSTM
[0.5545739531517029, 0.8016999959945679]
y_pred:  [7 2 1 0 4 1 4 9 5 9]
y_test:  [7 2 1 0 4 1 4 9 5 9]
'''

