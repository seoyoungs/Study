####keras23_LSTM3 카피해서
### LSTM 층을 2개 만들 ㅓㄳ

#####LSTM

import numpy as np
# 1. 데이터
x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
             [5,6,7], [6,7,8], [7,8,9],[8,9,10],
             [9,10,11], [10,11,12], 
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y= np.array([4,5,6,7,8,9,10,11,12,14,50,60,70])

#print('x.shape : ', x.shape) #(13,3)
#print('y.shape : ', y.shape) #(13,)

#x= x.reshape(13,3,1)
#print(x.shape[0]) #13
#print(x.shape[1]) #1
x=x.reshape(x.shape[0], x.shape[1],1)  #이제는 이렇게 표현 위에 참조

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1),return_sequences = True))
model.add(LSTM(20,activation='relu',return_sequences = False))
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
LSTM 1번 했을 때
1.292664885520935 (loss값)
[[80.406876]]--->epochs=180일때, predict값

LSTM 2번했을 때
0.09925279021263123
[[76.35107]

결론: 반복을 2번 하는 것보다
한번 LSTM 했을 때가 loss값은 낮고 
predict는 1번 했을 때가 80에 더 가깝다.

두개 이상 시계열인 LSTM 넣는다고 성능 좋아지진 않는다.
'''
