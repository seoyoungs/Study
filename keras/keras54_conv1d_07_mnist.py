import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('../data/npy/mnist_x_train.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
x_test= np.load('../data/npy/mnist_x_test.npy')
y_test =np.load('../data/npy/mnist_y_test.npy')

#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

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
from tensorflow.keras.layers import Dense, LSTM, Conv1D,Flatten
model=Sequential()
model.add(Conv1D(filters=15, kernel_size=1, padding='same',
                   strides=1 , input_shape=(7*7,16)))
model.add(Dense(8,activation='relu'))
model.add(Flatten())
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

conv1d
[0.1988861858844757, 0.9449999928474426]
'''