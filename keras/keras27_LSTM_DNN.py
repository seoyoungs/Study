###keras23_LSTM을 DNN으로 코딩
## 결과치 비교

## DNN으로 23번 파일보다 loss값 좋게 만들기

import numpy as np
# 1. 데이터
x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
             [5,6,7], [6,7,8], [7,8,9],[8,9,10],
             [9,10,11], [10,11,12], 
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y= np.array([4,5,6,7,8,9,10,11,12,14,50,60,70])

#print('x.shape : ', x.shape) #(13,3)
#print('y.shape : ', y.shape) #(13,)

###LSTM 으로 코딩
#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x= scaler.transform(x)

#x= x.reshape(13,3,1)
####re.shape응 하면 3차원이 된다. MinMaxScaler는 
#######

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential() #Sequential이 DNN이다
model.add(Dense(10, activation='relu', input_dim=3)) #x변경 됐음
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #LSTM있는 곳에 디폴트가 탄젠트인것, output은 linear임 
# 여기까지는 회귀값이다. 분류값 아님
#model.summary()

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x,y, epochs=180, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y)
print(loss)

x_pred =np.array([[50,60,70]]) #(3,) 행은 하나 -> (1,3,1)
x_pred= scaler.transform(x_pred)
result=model.predict(x_pred)
print(result)

##minmax하기 전에 reshape하기
'''
결과들 비교하기(같은 레이어로)
LSTM
1.292664885520935 (loss값)
[[80.406876]]--->epochs=180일때, predict값

SimpleRNN
0.10390414297580719
[[80.24279]]

GRU
0.04996849596500397
[[80.83638]]

DNN_MinMaxScaler
4.368030071258545
[[88.74426]]


---> SimpleRNN의 loss값이 가장 낮고 predict예측값이 80에 가장 가깝다.
'''


