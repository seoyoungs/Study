#네이밍룰-자바의 카멜케이스, 씨언어 언더바(_)
#import는 초반에 쓰면 좋음 model하기 전에만쓰면됨
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
x_train=np.array([1,2,3,4,5])
y_train=np.array([1,2,3,4,5]) 
#x,y개수 같게-데이터많을수록 훈련 더 잘됨

x_test=np.array([6,7,8]) #x의 평가 데이터
y_test=np.array([6,7,8])
#훈련데이터에는 x_train, y_train
#평가데이터는 x_test,y_test 따라서 평가데이터라서 훈련x
#train과 test로 나눴지만 실 데이터는 하나다.(1~8을 나눈것일뿐 하나이다.)

#2.모델 구성
model =Sequential()
# model =models.Sequential()
# model =keras.models.Sequential()
# from을 어떻게 설정하냐에 따라 모델구성 달라짐
model.add(Dense(5, input_dim=1, activation='relu'))
#activation 활성화 함수 relu 평타 85% 정확도 증가
model.add(Dense(25))
#activation을 안쓴것은 default(기본값)이 있다는 것 
# 위activation과 연관없음 안쓰면 linear로 자동으로 됨
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1)) # 마지막은 바꾸면 안됨

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
result = model.predict([9])
print("result : ", result)
