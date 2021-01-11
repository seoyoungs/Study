#텐서
#LSTM으로 모델링
#DENASE와 성능비교
#회귀 ---R2로 구하기 (acc는 분류에서)

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

'''
print(x[:5])
print(y[:10])
print(x.shape) #(442, 10)
print(y.shape) #(442, )
print(x.shape, y.shape) #(442, 10) - 10열개(10,0) (442,) -output1
print(np.max(x), np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)
'''

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.7, random_state=104)
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#print(x_train.shape) #(309, 10)
#print(x_test.shape)  #(133, 10)
x_train=x_train.reshape(309,10,1,1)
x_test=x_test.reshape(133,10,1,1)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1), 
                padding='same', strides=(1,1), input_shape=(10,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Dense(4, activation='relu'))
model.add(Flatten())
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=5, validation_split=0.3, verbose=1)

#4. 평가예측
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
print('loss, mae : ', loss, mae)
y_predict=model.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

'''
LSTM 전
loss, mae :  2456.418701171875 40.79722595214844
RMSE :  49.56227114192523
R2 : 0.5754409282584073
LSTM 한 후
loss, mae :  28272.857421875 149.956787109375
RMSE :  168.14533939572718
R2 : -3.8865842948730656

CNN한후
loss, mae :  2668.55029296875 42.008934020996094
RMSE :  51.658009920239536
R2 : 0.5387768800883477
'''