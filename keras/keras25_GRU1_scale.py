### 코딩 실습
### predict 80만들기

import numpy as np
# 1. 데이터
x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
             [5,6,7], [6,7,8], [7,8,9],[8,9,10],
             [9,10,11], [10,11,12], 
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y= np.array([4,5,6,7,8,9,10,11,12,14,50,60,70])

x=x.reshape(13,3,1)

#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1))) #x변경 됐음
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
결과
0.04996849596500397
[[80.83638]]
SimpleRNN보다 결과값 좋다

연산이 적다. (GRU)=LSTM-cell State 그럼 360이여야 하는데 왜 390일까?
이런 이유 찾기
((input_dim+output))*output+output+output)*3 = 390
((n+m)*m+m+m)*3 
((1+10)*10+10+10)*3=390

activation Default(디폴트): tangent 
'''