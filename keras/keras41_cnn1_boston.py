######### cnn으로 구성
######## 이게 2차원이니 알아서 4차원으로 늘려서 하기
#####boston은 회귀 이므로 R2만 구하기

import numpy as np #gpu단순연산에 좋다.

from sklearn.datasets import load_boston
#사이킷런에서 데이터 기본 제공

#1. 데이터
dataset= load_boston()
x=dataset.data
y=dataset.target

#print(x.shape) #(506. 13)
#print(y.shape) #(506)

x=x.reshape(x.shape[0], x.shape[1],1,1) #=x.reshape(-1,13,1,1)
print(x.shape) #((506, 13, 1, 1))

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model=Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1), 
                padding='same', strides=(1,1), input_shape=(13,1,1)))
model.add(Dense(25, activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=16,
          validation_split=0.2, verbose=1)

#4. 평가예측
loss=model.evaluate(x_test, y_test,batch_size=16)
print('loss:', loss)
y_pred = model.predict(x_test)
#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

'''
 LSTM  전
결과
loss: 19.869380950927734
mae : 3.2330851554870605
RMSE :  4.457507891337804
R2 :  0.7622795125940092
LSTM 후
loss: 15.670780181884766
mae : 2.846346378326416
RMSE :  3.958633703813784
R2 :  0.8125122028343794

CNN적용 후
loss: 15.950458526611328
R2 :  0.8091661202176189
'''

