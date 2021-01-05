##keras_LSRM3_scale 함수형으로 코딩
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

inputs=Input(shape=(3,1))
lstm_layer = LSTM(13, activation='relu')(inputs)
aaa = Dense(20, activation='relu')(lstm_layer)
aaa=Dense(10)(aaa)
aaa=Dense(5)(aaa)
outputs=Dense(1)(aaa)
#차례대로 앞에 있던게 맨 뒤로 온다. input을 뒤에 명시
model=Model(inputs, outputs)
#model.summary()

'''
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) #x변경 됐음
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #LSTM있는 곳에 디폴트가 탄젠트인것, output은 linear임 
# 여기까지는 회귀값이다. 분류값 아님
model.summary()
'''
#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x,y, epochs=10, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y)
print(loss)

x_pred =np.array([50,60,70]) #(3,) 행은 하나 -> (1,3,1)
x_pred= np.reshape(x_pred,(1,3,1))

result=model.predict(x_pred)
print(result)

###input shape가 2차원으로 되있을 것
### 전처리할 때 predict도 transfrom 하기
