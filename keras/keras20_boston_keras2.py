#주말과제
# 2개의 파일만들어라 keras20_boston_keras1.py keras20_boston_keras2.py
#1. EarliStopping을 적용안한 최고의 모델
#2. EarliStopping을 적용한 최고의 모델
import numpy as np
from tensorflow.keras.datasets import boston_housing
#이걸로 만들어라라라ㅏ라라
#tensorflow에서 제공하므로 sklearn과 아예 다르다. x,y 할당법 찾기
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

'''
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(len(x_train), len(x_test))
print(x_train[0])
print(y_train[0])
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
x_train -= x_mean
x_train /= x_std
x_test -= x_mean
x_test /= x_std
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std
'''
print(x_train[0])
print(y_train[0])

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model=Sequential()
model.add(Dense(15, activation='relu', input_dim=13))
model.add(Dense(10, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
#model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=250, batch_size=10,
        validation_split=0.2, verbose=1, callbacks=[early_stopping])

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test, batch_size=4)
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
earliststopping
loss, mae :  0.2743679881095886 0.33615097403526306
RMSE :  0.5238014693073252
R2 :  0.721088862526471
'''


