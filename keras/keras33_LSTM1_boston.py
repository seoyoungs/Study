#사이킷런
#LSTM으로 모델링
#DENASE와 성능비교
#회귀

import numpy as np #gpu단순연산에 좋다.

from sklearn.datasets import load_boston
#사이킷런에서 데이터 기본 제공

#1. 데이터
dataset= load_boston()
x=dataset.data
y=dataset.target

#print(x.shape) #(506. 13)
#print(y.shape) #(506)

x= x.reshape(506,13,1)

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input, LSTM

model=Sequential()
model.add(LSTM(12, activation='relu', input_shape=(13,1))) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(4, activation='relu'))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_split=0.2, verbose=1)

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test,batch_size=5)
print('loss:', loss)
print('mae :', mae)
#print('loss, mae :', loss, mae)
y_predict=model.predict(x_test) 
#print(y_predict)
#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict)) 

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
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
'''