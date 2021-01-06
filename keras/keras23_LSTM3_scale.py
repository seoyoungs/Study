###LSTM 으로 코딩

import numpy as np
# 1. 데이터
x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
             [5,6,7], [6,7,8], [7,8,9],[8,9,10],
             [9,10,11], [10,11,12], 
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y= np.array([4,5,6,7,8,9,10,11,12,14,50,60,70])

#print('x.shape : ', x.shape) #(13,3)
#print('y.shape : ', y.shape) #(13,)

x= x.reshape(13,3,1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) #x변경 됐음
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #LSTM있는 곳에 디폴트가 탄젠트인것, output은 linear임 
# 여기까지는 회귀값이다. 분류값 아님
model.summary()

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x,y, epochs=180, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y)
print(loss)

x_pred =np.array([50,60,70]) #(3,) 행은 하나 -> (1,3,1)
x_pred= x_pred.reshape(1,3,1)
result=model.predict(x_pred)
print(result)

'''
LSTM
model.add(LSTM(10, activation='relu', input_shape=(3,1))) #x변경 됐음
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
결과
predict가 80이 나오는게 목적이다.
0.02959594875574112 (loss값)
[[80.317726]] --->epochs=150, predict값1
0.04683738201856613 (loss값)
[[79.98581]] --->epochs=150, predict값2
1.292664885520935 (loss값)
[[80.406876]]--->epochs=180일때, predict값
'''

