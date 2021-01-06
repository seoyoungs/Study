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
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1))) #x변경 됐음
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
SimpleRNN
0.10390414297580719
[[80.24279]]
통상적으로 LSTM이 loss값 더 낮게 나온다.
결과값 좋다

summary를 보면 파라미터를
(3,1)==> input_length=3, input_dim=1
(None, 10)  => output=10
(n+m+1)*m  = (1+10+1)*10 =120
*****LSTM이랑 다른 점 : 되돌아 오는 게 없다.

activation Default(디폴트): hyperbolic tangent 

SimpleRNN 의 문제점
반복이 LSTM과 다르게 없다
그럼 연속성이 부족한 데이터가 많은 자료일 때
반복이 없어 앞쪽에 있는 자료들에게 연산의 영향이 미치지 못한다.
그래서 LSTM을 사용해 앞쪽에 있는 것도 같이 고려하는 것이다.

 LSTM은 계산이 좋지만 느리다.

'''
