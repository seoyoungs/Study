#hist를 이용해 그래프를 그려보시오
#loss, val_loss
#### 회귀모델 acc적용 x

import numpy as np #gpu단순연산에 좋다.

from sklearn.datasets import load_boston
#사이킷런에서 데이터 기본 제공

#1. 데이터
dataset= load_boston()
x=dataset.data
y=dataset.target
#print(x.shape) #(506, 13)
#print(y.shape)  #(506,)
#print("==============")
#print(x[:5])
#print(y[:10])

#print(np.max(x), np.min(x)) #711.0 0.0(이렇게 1보다 클수 있다. 전처리방법 다양) 

#데이터 전처리 libarary
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) #---> x_train으로 변경
x=scaler.transform(x)
print(np.max(x), np.min(x)) #711.0 0.0 -> 1.0 0.0
print(np.max(x[0]))
#근데 이것도 문제가 있다.--대체 why....?

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

model=Sequential()
model.add(Dense(12, activation='relu', input_dim=13)) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(4, activation='relu'))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#얼리 스탑핑 적용
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = model.fit(x,y, epochs=500, batch_size=16, verbose=1, 
         validation_split=0.2, callbacks=[es])

print(hist)
print(hist.history.keys())
#print(hist.history['loss']) ###로스값이 차례대로 줄어드는 것을 볼 수 있다.
##그림을 loss값을 토대로 그릴 예정
####그래프####
import matplotlib.pyplot as plt
#만약 plt.plot(x,y)하면 x,y 값이 찍힌다.
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'])
plt.show()
