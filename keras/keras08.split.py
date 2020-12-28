from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#np.array()
#array()

#1. data - 데이터 자르기(슬라이싱: 통데이터때 활용할 줄 알아야함)
x=np.array(range(1,101)) # 1 ~ 100   # x = np.array(range(100))        # 0 ~ 99
#print(x) #이렇게 하면 뒷자리 101에서-1 되서 100까지 나온다.

y= np.array(range(101,201))

x_train=x[:60] 
# : 이 뜻은 처음부터 60까지 라는 뜻-> 이거는 랭지와 다르게 순서0~59번째까지 이므로 값은1~60이다.
x_val=x[60:80] #79번째 값이 80, 59번째 값이 60, 따라서 61~80이다.
x_test=x[80:]  #81~100 ->x의 끝값이 100까지 이므로
#즉 위에는 리스트의 슬라이싱이라고 한다.

y_train=y[:60] 
y_val=y[60:80]
y_test=y[80:] 
#밑에 validation_split이미 했으므로 validation_data로 바꿔준다.

#따옴표 원하는 구간 위,아래 3개씩 하면 주석된다.

x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])

x_test= array([11,12,13,14,15]) #validation 넣을 수 있다.
y_test= array([11,12,13,14,15]) #from numpy import array 언급 위에 했으으로 array만 해도된다.

x_pred=array([16,17,18]) #y값 알고 싶어서 이것만 한다.

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100,batch_size=1, validation_data=(x_val, y_val))
# x_train, y_train을 validation_split=0.2로 20%를 나누어(쪼개어)쓰겠다. train에 80%준것

# 4. evaluate, predict
result = model.evaluate(x_test, y_test, batch_size=1)
#result에는 loss(-> mse), matrix(-> mae)가 들어간다.
# print("result : ", result)
print("mse, mae : ", result)
y_predict=model.predict(x_test)
#print("y_predict :", y_predict)

#사이킷런
from sklearn.metrics import mean_squared_error
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임

print("RMSE : ", RMSE(y_test,y_predict)) #MSE,mae RMSE가 결과값으로 나온다.
#print("mse: ",mean_squared_error(y_test, y_predict))
print("mse: ",mean_squared_error(y_predict, y_test))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
