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

x_pred=array([16,17,18]) #y값 알고 싶어서 이것만 한다.

#2. modeling
model=Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, training
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100,batch_size=1, validation_split=0.2)
# x_train, y_train을 validation_split=0.2로 임의적으로 20%를 나누어(쪼개어)쓰겠다. 
# 즉, train 10개 중에 8개쓰겠다. (validation data 대신 쓴것)
#2.3번 모두 파라미터 튜닝에 해당된다.

# 4. evaluate, predict
result = model.evaluate(x_test, y_test, batch_size=1)
#result에는 loss(-> mse), matrix(-> mae)가 들어간다.
print("result : ", result)

y_pred=model.predict(x_pred)
print("y_predict :", y_pred)
