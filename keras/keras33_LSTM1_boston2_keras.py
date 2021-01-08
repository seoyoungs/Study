#사이킷런
#LSTM으로 모델링
#DENASE와 성능비교
#회귀

import numpy as np
from tensorflow.keras.datasets import boston_housing


#tensorflow에서 제공하므로 sklearn과 아예 다르다. x,y 할당법 찾기

#방법1
#(x_train, y_train), (x_test, y_test) = boston_housing.load_data() 

#방법2
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

'''
print(x_train.shape) #(404, 13)
print(y_train.shape) #(404,)
print(x_test.shape) #(102, 13)
print(y_test.shape) #(102,)
'''

x_train=x_train.reshape(404, 13,1)
x_test=x_test.reshape(102,13,1)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                    train_size=0.8, shuffle=True, random_state=311)

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
'''

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

model=Sequential()
model.add(LSTM(15, activation='relu', input_shape=(13,1)))
model.add(Dense(11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
#model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=5, 
          validation_split=0.3, verbose=1)

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test, batch_size=5)
print('loss, mae : ', loss, mae)
y_predict= model.predict(x_test)


#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ',RMSE(y_test,y_predict))

#R2
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('R2 : ', r2)

'''
earliststopping 적용 x
loss, mae :  0.2344101369380951 0.3044426739215851
RMSE :  0.48405710279491826
R2 :  0.7618088471455072
loss, mae :  518.4482421875 518.4482421875
RMSE :  22.76945806251415
R2 :  -5.2280664539107
'''