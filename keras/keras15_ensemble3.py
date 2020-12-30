# 다:다 앙상블 만들기
# 인풋2, 아웃풋3 앙상블 만들기
#x1, x2, y1, y2, y3

import numpy as np

#앙상블 모델이 2개
#1.데이터
x1=np.array([range(100),range(301, 401), range(1,101)])
y1=np.array([range(711,811), range(1,101), range(201,301)])#각100개

x2=np.array([range(101, 201), range(411,511), range(100, 200)])
y2=np.array([range(501, 601), range(711, 811), range(100)])

y3=np.array([range(601, 701), range(811, 911), range(1100,1200)])
#3행 100열 모델

x1= np.transpose(x1)
x2= np.transpose(x2)
y1= np.transpose(y1)
y2= np.transpose(y2)
y3= np.transpose(y3)

#pred
x_pred2=np.array([100,302,101]) #행렬 바꾸기1
x_pred3=np.array([100,302,101]) #행렬 바꾸기1
x_pred2=x_pred2.reshape(1,3)
x_pred3=x_pred2.reshape(1,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test =train_test_split(
    x1, y1, shuffle=False, train_size=0.8) #shuffle=False순서대로
x2_train, x2_test, y2_train, y2_test =train_test_split(
    x2, y2, shuffle=False, train_size=0.8)
y3_train, y3_test =train_test_split(
        y3, shuffle=False, train_size=0.8)

#모델링 할 때 Sequential로 하면 순서대로 한다는 뜻이다
#근데 이 두모델은 수평적 관계라 순차적인 순위를 매기기 힘듬
#그래서 다른 모델링 쓸 예정

#2. 모델구성
from tensorflow.keras.models import Sequential, Model 
#Model은 함수형으로 입력하겠다는 뜻이다
from tensorflow.keras.layers import Dense, Input

#모델 1
input1= Input(shape=(3,))
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)
#output1=Dense(3)(dense1)

#모델 2
input2= Input(shape=(3,))
dense2=Dense(10, activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
#output2=Dense(3)(dense2)

#모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
#from keras.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2]) #모델 1,2 병합
middle1=Dense(30)(merge1) #모델 합친후에도 layer추가 가능
middle1=Dense(10)(merge1)
middle1=Dense(10)(merge1) 

#데이터 합친 것 다시 나누기(100,3)2개니까 (100,6)된것 다시 나누기
#모델 분기1
output1= Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(3)(output1) #여기까지가 y1

#모델 분기2
output2= Dense(30)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(3)(output2) #여기까지가 y2

#모델 분기3
output3= Dense(30)(middle1)
output3=Dense(7)(output3)
output3=Dense(7)(output3)
output3=Dense(3)(output3) #여기까지가 y3

#모델 선언 (함수형으로)
model = Model(inputs=[input1,input2], 
              outputs=[output1, output2, output3]) #두개 이상은 list로 묶는다.
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train],
           epochs=10, batch_size=1, validation_split=0.2, verbose=1) 
#모델 2개이므로 원래 x_train, y_train 넣었는데 2개이므로 list사용
#모델 2개 이므로 list([])로 묶는다.

#4. 평가, 예측
loss= model.evaluate([x1_test, x2_test], 
         [y1_test, y2_test, y3_test], batch_size=1)
print(loss)
'''
#concatenate 뜻
metrics=['mse']로 했을 때
[2737.472412109375, 1091.6365966796875, 1645.8359375, 1091.6365966796875, 1645.8359375] 
[1091.6365966796875, 1645.8359375]을 합한 값은 2384.06982421875
loss의 mse와 metrics의 mae값이 같다
concatenate값1개, loss값 2개, metrics값2개 해서 총 5개 값이 나온다.
'''
print("model.metrics_names :", model.metrics_names)
# model.metrics_names : ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']
#이걸보면 print(loss) 했을 때 값 알 수 있다. summary 참조

y1_predict, y2_predict, y3_predict =model.predict([x1_test, x2_test]) 
'''
print("==============")
print("y1_predict : ",y1_predict)
print("==============")
print("y2_predict :", y2_predict)
print("==============")
#test_size가 20개 이므로 20개나온다.
'''
#사이킷런 RMSE
from sklearn.metrics import mean_squared_error
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임

RMSE1=RMSE(y1_test, y1_predict)
RMSE2=RMSE(y2_test, y2_predict)
RMSE3=RMSE(y3_test, y3_predict)
RMSE=(RMSE1+RMSE2+RMSE3)/3
print('RMSE1 : ', RMSE1)
print('RMSE2 : ', RMSE2)
print('RMSE3 : ', RMSE3)
print('RMSE :', RMSE)

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
R2_1 = r2_score(y1_test, y1_predict)
R2_2 = r2_score(y2_test, y2_predict)
R2_3 = r2_score(y3_test, y3_predict)
R2=(R2_1+R2_2+R2_3)/2
print('R2_1 : ', R2_1)
print('R2_2 : ', R2_2)
print('R2_3 : ', R2_3)
print('R2 : ', R2)


#다만들면 predict의 일부값을 출력하시오
y_pred=model.predict([x_pred2,x_pred2])
print(y_pred)