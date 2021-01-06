###한개의 아웃풋
########실습 : 앙상블 모델을 만드시오
####predict 85 근사치 나오게 하기
###### 앙상블 할 때 아웃푹 차원 잘 쓰기

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
x1_predict= np.array([55,65,75]) #(3,) ->Dense (1,3)--> LSTM일때(1,3,1)
x2_predict=np.array([65,75,85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
x1_predict = x1_predict.reshape(1, 3, 1) #(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1) #(1, 3, 1)

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate

#모델 1
inputs1=Input(shape=(3,1))
aaa1 = LSTM(13, activation='relu')(inputs1)
aaa1 = Dense(20, activation='relu')(aaa1)
aaa1=Dense(10, activation='relu')(aaa1)
aaa1=Dense(1, activation='relu')(aaa1)
#outputs1=Dense(1)(aaa1)
#model=Model(inputs=inputs1, outputs=outputs1)

#모델 2
inputs2=Input(shape=(3,1))
aaa2 = LSTM(13, activation='relu')(inputs2)
aaa2= Dense(20, activation='relu')(aaa2)
aaa2=Dense(10, activation='relu')(aaa2)
aaa2=Dense(12, activation='relu')(aaa2)
#outputs2=Dense(1)(aaa2)
#model=Model(inputs=inputs2, outputs=outputs2)

#모델 병합 / concatenate
merge1 = concatenate([aaa1,aaa2]) #모델 1,2 병합
middle1=Dense(30)(merge1) #모델 합친후에도 layer추가 가능
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

#모델 분기1
output1= Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(7)(output1)
output1=Dense(1)(output1) ###output은 1개 이니 1로 

model = Model(inputs=[inputs1,inputs2], 
              outputs=output1) #두개 이상은 list로 묶는다.
#model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y1,
           epochs=10, batch_size=3, validation_split=0.2, verbose=1) 
#모델 2개이므로 원래 x_train, y_train 넣었는데 2개이므로 list사용
#모델 2개 이므로 list([])로 묶는다.

#4. 평가, 예측
loss= model.evaluate([x1, x2], y1, batch_size=3)
print(loss)
y_pred= model.predict([x1_predict, x2_predict])
print('y_pred: ', y_pred)

'''
결과
15.657464027404785
y_pred:  [[77.87596]]

'''