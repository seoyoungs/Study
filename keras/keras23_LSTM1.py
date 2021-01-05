#####LSTM
# 1. 데이터
import numpy as np

x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y=np.array([4,5,6,7])


print('x.shape : ', x.shape) #(4,3)
print('y.shape : ', y.shape) #(4,)

x= x.reshape(4,3,1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) #x변경 됐음
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1)) #LSTM있는 곳에 디폴트가 탄젠트인것, output은 linear임 
# 여기까지는 회귀값이다. 분류값 아님
model.summary()


#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y)
print(loss)

x_pred=np.array([5,6,7]) #(3,) 행은 하나 -> (1,3,1)
x_pred= x_pred.reshape(1,3,1)

result=model.predict(x_pred)
print(result)

'''
model.summary()를 하면 
LSRM 파라미터 개수의 공식은
4*(n+m+1)*m 이다
4*(1+1+10)*10
n: input_dim
m: output (여기서 10)
 input_shape=(3,1) ====> input_length=3, input_dim=1
LSTM 디폴트 값은 탄젠트
'''