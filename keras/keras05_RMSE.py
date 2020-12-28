from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#np.array()
#array()

#1. data
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])

x_test= array([11,12,13,14,15]) #validation 넣을 수 있다.
y_test= array([11,12,13,14,15]) #from numpy import array 언급 위에 했으으로 array만 해도된다.

x_pred=array([16,17,18]) #y값 알고 싶어서 이것만 한다. 미래

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100,batch_size=1, validation_split=0.2)
# x_train, y_train을 validation_split=0.2로 20%를 나누어(쪼개어)쓰겠다.
# metrics=['mae']나 mse로 하기 위해서는 데이터를 분류해야한다. ex) 남녀로 나눈다.


# 4. evaluate, predict
result = model.evaluate(x_test, y_test, batch_size=1)
#result에는 loss(-> mse), matrix(-> mae)가 들어간다.
print("mse, mae : ", result)

#print("result : ", result)

y_predict=model.predict(x_test)
#print("y_predict :", y_predict)

#사이킷런
from sklearn.metrics import mean_squared_error
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# 결국 y_predict는 x_test를 예상하는 것과 같다.
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임

print("RMSE : ", RMSE(y_test,y_predict)) #MSE,mae RMSE가 결과값으로 나온다.
#print("mse: ",mean_squared_error(y_test, y_predict))
print("mse: ",mean_squared_error(y_predict, y_test))

