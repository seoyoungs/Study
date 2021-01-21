##실습 
# dropout 적용

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target


#print(x[:5])
#print(y[:10])
#print(x.shape, y.shape) #(442, 10) - 10열개(10,0) (442,) -output1
#print(np.max(x), np.min(x))
#print(dataset.feature_names)
#print(dataset.DESCR)

#x = x/0.198787989657293

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.3, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler =StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

inputs1=Input(shape=(10,))
aaa= Dense(15, activation='relu')(inputs1)
drop =Dropout(0.2)(aaa)
aaa= Dense(20, activation='relu')(drop)
aaa= Dense(10, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
outputs=Dense(1)(aaa)
model= Model(inputs= inputs1, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=10, 
           validation_split=0.2, verbose=1)

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

'''
파라미터는 그대로 이다. dropout은 layer가 아니므로 적용안한다
실질적으로 train 할 때만 dropout적용
test시에는 그대로 레이어 다쓴다.
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 10)]              0
_________________________________________________________________
dense (Dense)                (None, 15)                165       
_________________________________________________________________
dropout (Dropout)            (None, 15)                0
_________________________________________________________________
dense_1 (Dense)              (None, 20)                320
_________________________________________________________________
dense_2 (Dense)              (None, 10)                210
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 786
Trainable params: 786
Non-trainable params: 0
_________________________________________________________________

dropout적용 전
loss, mae :  2882.239013671875 43.667938232421875
RMSE :  53.686493167103734
R2 : 0.5026237627109889

dropout적용 후
loss, mae :  2922.3046875 43.70711135864258
RMSE :  54.05834607922155
R2 : 0.49570987019044643

StandardScaler --> 더 떨어짐
loss, mae :  3069.0283203125 45.43492889404297
RMSE :  55.39881194238311
R2 : 0.47039038414313894
'''