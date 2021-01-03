#실습: 19_1,2,3,4,5, EarliStopping까지
#총 6개 파일을 만드시오

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target


#print(x[:5])
#print(y[:10])
#print(x.shape, y.shape) #(442, 10) - 10열개(10,0) (442,) -output1
print(np.max(x), np.min(x))
#print(dataset.feature_names)
#print(dataset.DESCR)

#x = x/0.198787989657293
#데이터 전처리 libarary
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) #---> x_train으로 변경
x=scaler.transform(x)

print(np.max(x), np.min(x)) #711.0 0.0 -> 1.0 0.0
print(np.max(x[0]))


#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.3, random_state=66)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

inputs1=Input(shape=(10,))
aaa= Dense(15, activation='relu')(inputs1)
aaa= Dense(8, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
aaa= Dense(3, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
outputs=Dense(1)(aaa)
model= Model(inputs= inputs1, outputs=outputs)
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=0)

#4. 평가예측
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
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
