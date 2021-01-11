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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath1 = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=5)
cp =ModelCheckpoint(filepath='modelpath1', monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1,
                 validation_split=0.2,callbacks=[es, cp])

#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=32)
print('loss :', loss)
y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

####시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) #가로세로 #이거 한 번 만 쓰기
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

#plt.title('손실비용') #한글깨짐 오류 해결할 것 과제1
plt.title('Cost loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

#plt.title('정확도')
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

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