import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], -1))/255.
x_test = x_test.reshape((x_test.shape[0], -1))/255.
print(x_train.shape)
'''
# 학습 과정을 단축시키기 위해 학습 데이터의 1/3만 활용한다
from sklearn.model_selection import train_test_split
x_train, _ , y_train, _ = train_test_split(x_train, y_train, 
                          test_size = 0.67, random_state = 7)
'''
#다중분류 y원핫코딩

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model=Sequential()
model.add(Dense(units=12, activation='relu', input_shape=(28*28,)))
model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=15, 
           validation_split=0.2, batch_size=32,verbose=1)

#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=32)
print('loss :', loss)
y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

'''
CNN
[0.3294849693775177, 0.8870000243186951]
y_pred:  [9 2 1 1 6 1 4 6 5 7]
y_test:  [9 2 1 1 6 1 4 6 5 7]

DNN
loss : [0.45912548899650574, 0.8371999859809875]
y_pred:  [9 2 1 1 6 1 4 6 5 7]
y_test:  [9 2 1 1 6 1 4 6 5 7]
'''