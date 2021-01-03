# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8,
                                                  shuffle=True, random_state=104)
x_train, x_val, y_train, y_val= train_test_split(x,y, 
                           shuffle=True, train_size=0.7, random_state=104)

#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test= scaler.transform(x_test)
x_val= scaler.transform(x_val)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model=Sequential()
model.add(Dense(12, activation='relu', input_dim=10))
model.add(Dense(4, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

'''
inputs1=Input(shape=(10,))
dense1=Dense(15, activation='relu')(inputs1)
dense1=Dense(7, activation='relu')(dense1)
outputs1=Dense(1)(dense1)
model=Model(inputs=inputs1, outputs=outputs1)
'''
#컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=5, 
          validation_data=(x_val, y_val), verbose=1)

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



