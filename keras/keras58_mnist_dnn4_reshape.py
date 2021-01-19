#keras45 카피했음
# ##요점 4차원도 dense 모델이 가능하다.
####2.모델링에서 reshape넣기

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#다중분류이므로 원핫코딩, 전처리 해야한다.
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.


y_train = x_train
y_test = x_test
#(6000,28)

print(y_train.shape) #(60000, 28, 28, 1)
print(y_test.shape) #(10000, 28, 28, 1)


#2. 모델 구성 -->conv2d도 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Reshape

model =Sequential()
model.add(Dense(64,input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dense(5))
# model.add(Dropout(0.5))
model.add(Flatten()) #만약 4차원으로 출력되면 이것도 괜찮다.
model.add(Dense(64)) 
model.add(Dense(784, activation='relu')) ###reshape가 되려면 위에가 28*28=784
model.add(Reshape((28,28,1)))  #__init__() takes 2 positional이 에러나면 괄호넣기
#reshape 모양을 넣을때 튜플이라는 데이터 형태로 넣어야해서 괄호를 하나 더 쓴다.
model.add(Dense(1)) #y값 3개이다(0,1,2)
model.summary() #(28,28,1) 나오게 하려면 마지막 덴스1로

'''
#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#ReduceLROnPlateau ---> learning late감소하겠다.
modelpath = '../data/modelCheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=6)
cp =ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factor = 0.5, verbose=1)
#ReduceLROnPlateau----3번까지 개선안되는 거 되는데 그 이후로는 0.5로 깍겠다.
model.compile(loss='mse',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1,
                 validation_split=0.5,callbacks=[es, cp, reduce_lr])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])

y_pred= model.predict(x_test) #  x_test: #(10000, 28, 28, 1)
print(y_pred[0]) #한개만 출력
print(y_pred.shape) #(10000, 28, 28, 1)
'''