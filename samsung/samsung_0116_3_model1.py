import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


x1= np.load('../data/npy/sam_new2.npy',allow_pickle=True)[0]
y1= np.load('../data/npy/sam_new2.npy',allow_pickle=True)[1]

x2= np.load('../data/npy/kodex1.npy',allow_pickle=True)[0]
y2= np.load('../data/npy/kodex1.npy',allow_pickle=True)[1]

y1= np.delete(y1, 1, axis = 0)
y2= np.delete(y2, 1, axis = 0)
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test =train_test_split(
    x1, y1, shuffle=False, train_size=0.8) #shuffle=False순서대로
x2_train, x2_test, y2_train, y2_test =train_test_split(
    x2, y2, shuffle=False, train_size=0.8)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1], 1)
x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1], 1)
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1], 1)
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1], 1)
x_pred=x1[:1, :5]
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
print(x_pred.shape)

'''
#2. 모델구성 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input, LSTM

#모델 1
inputs1=Input(shape=(6,1))
lstm_layer = LSTM(13, activation='relu')(inputs1)
aaa1 = Dense(20, activation='relu')(lstm_layer)
aaa1=Dense(10)(aaa1)
aaa1=Dense(1)(aaa1)
#outputs1=Dense(1)(aaa1)
#model=Model(inputs=inputs1, outputs=outputs1)

#모델 2
inputs2=Input(shape=(6,1))
aaa2= LSTM(20, activation='relu')(inputs2)
aaa2=Dense(10,  activation='relu')(aaa2)
aaa2=Dense(1)(aaa2)
#outputs2=Dense(1)(aaa2)
#model=Model(inputs=inputs2, outputs=outputs2)

#모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([aaa1,aaa2]) #모델 1,2 병합
middle1=Dense(30)(merge1) #모델 합친후에도 layer추가 가능
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

#모델 분기1
output1= Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(7)(output1)
output1=Dense(2)(output1) 

#모델 선언 (함수형으로)
model = Model(inputs=[inputs1,inputs2], 
              outputs=[output1]) #두개 이상은 list로 묶는다.
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train],
           epochs=10, batch_size=1, validation_split=0.2, verbose=1) 
#모델 2개이므로 원래 x_train, y_train 넣었는데 2개이므로 list사용
#모델 2개 이므로 list([])로 묶는다.

#4. 평가, 예측
loss= model.evaluate([x1_test, x2_test], [y1_test, y2_test]
                      ,batch_size=1)
print(loss)


x_pred2=np.array([91800,88000,88000,33117980,2947682,7510662]) #행렬 바꾸기1
x_pred2=x_pred2.reshape(1,6,1)
y_pred=model.predict(x_pred2)
print(y_pred)
'''

