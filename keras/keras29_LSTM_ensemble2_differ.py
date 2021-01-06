###한개의 아웃풋
####2개의 모델을 하나는 LSTM, 하나는 dense로
#앙상블 구현
#29_1번과 성능비교


import numpy as np
import numpy as array
#1. 데이터
x1= np.array([[1,2,3],[2,3,4],[3,4,5], [4,5,6], 
           [5,6,7],[6,7,8], [7,8,9], [8,9,10],
           [9, 10,11], [10,11,12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])
x2= np.array([[10, 20, 30], [20, 30, 40], [30, 40,50],[40, 50, 60],
          [50, 60, 70], [60,70, 80], [70,80, 90], [80, 90,100],
          [90, 100, 110], [100, 110, 120],
          [2,3,4], [3,4,5],[4,5,6]])
y1= np.array([4,5,6,7,8,9,10,11,12,13,50, 60, 70])

x1=x1.reshape(13, 3,1)
x2=x2.reshape(13, 3)

'''
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test =train_test_split(
    x1,x2, y1, shuffle=False, train_size=0.8)
'''
#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input, LSTM

#모델 1
inputs1=Input(shape=(3,1))
lstm_layer = LSTM(13, activation='relu')(inputs1)
aaa1 = Dense(20, activation='relu')(lstm_layer)
aaa1=Dense(10)(aaa1)
aaa1=Dense(1)(aaa1)
#outputs1=Dense(1)(aaa1)
#model=Model(inputs=inputs1, outputs=outputs1)

#모델 2
inputs2=Input(shape=(3,))
aaa2= Dense(20, activation='relu')(inputs2)
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
output1=Dense(1)(output1) 

model = Model(inputs=[inputs1,inputs2], 
              outputs=output1) #두개 이상은 list로 묶는다.
#model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1, x2], y1,
           epochs=10, batch_size=1, validation_split=0.2, verbose=1) 
#모델 2개이므로 원래 x_train, y_train 넣었는데 2개이므로 list사용
#모델 2개 이므로 list([])로 묶는다.

#4. 평가, 예측
loss= model.evaluate([x1, x2], y1, batch_size=1)
print(loss)

x1_predict= np.array([55,65,75]) #(3,) ->Dense (1,3)--> LSTM일때(1,3,1)
x2_predict=np.array([65,75,85])

x1_predict= x1_predict.reshape(1,3,1)
x2_predict= x2_predict.reshape(1,3,1)
result= model.predict([x1_predict, x2_predict])
print('result:',result)
