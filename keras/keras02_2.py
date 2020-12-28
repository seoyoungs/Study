import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras import model로 하면 
# keras의 model을 불러들이는 것
# 2에서 모델구성 model =models.Sequential() 이렇게 해야 에러 안남
# from tensorflow import keras로 하면 
# 2에서 모델구성 model =keras.models.Sequential() 이렇게 해야 에러 안남

#1.데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([2,4,6,8,10,12,14,16,18,20])
#x,y개수 같게-데이터많을수록 훈련 더 잘됨

x_test=np.array(list(range(101,110)))
y_test=np.array(list(range(111,120)))
#x_test=np.array([101,102,103,104,105,106,107,108,109,110]) #x의 평가 데이터
#y_test=np.array([111,112,113,114,115,116,117,118,119,120]) #y의 평가 데이터
#훈련데이터에는 x_train, y_train
#평가데이터는 x_test,y_test 따라서 평가데이터라서 훈련x

x_predict=np.array([111,112,113]) #훈련된 프레딕트 값

#2.모델 구성
#객체 생성
model =Sequential()
# model =models.Sequential()
# model =keras.models.Sequential()
# from을 어떻게 설정하냐에 따라 모델구성 달라짐
model.add(Dense(1500, input_dim=1, activation='sigmoid'))
#activation 활성화 함수
#activation을 안쓴것은 default(기본값)이 있다는 것 
# 위activation과 연관없음 안쓰면 linear로 자동으로 됨
model.add(Dense(1000, input_dim=1, activation='linear')) #linear 선형관계
#linear이라 선형 회귀 즉, 회귀 식이다.
model.add(Dense(800,input_dim=1, activation='linear')) 
model.add(Dense(600,input_dim=1, activation='linear'))
model.add(Dense(300,input_dim=1, activation='linear'))
model.add(Dense(1,input_dim=1, activation='linear')) # 마지막은 바꾸면 안됨

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=256) 

#4.평가 예측
loss=model.evaluate(x_test, y_test, batch_size=256)
print('loss : ',loss)
pred = model.predict([x_predict])
print('pred :', pred)
