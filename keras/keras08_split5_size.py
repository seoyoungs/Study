#실습 validation_data를 만들것
#train_test_split을 사용할 것
#20%면 validation data는 16개다 train이80개이니 이중 16개를 validation으로한다.
#그럼 100개 데이터 중 train=64, val=16, test=20 이렇게 나뉜다.
#실습 train_size와 test_size 비교후 이해하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array(range(1,101)) #1~100
y=np.array(range(1,101)) #101~200

from sklearn.model_selection import train_test_split
#우리 목적 train, test 분리(split)하는 것
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                   train_size=0.7, test_size=0.2, shuffle=False)
                                   #train_size=0.7, test_size=0.2, shuffle=False
                                   #train_size=0.9, test_size=0.2, shuffle=False
                                   #위 두가지경우에대해 확인 후 정리할 것
                                   #한개는 1보다 작고 1개는 1보다 크다.
#x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.6, shuffle=True)
#이렇게 하면 train_size=0.6이면 x_train에 60, test에는 40퍼센트, 랜덤으로 60개 값이 추출된다.
# 이것말고 랜덤하게 안하려면 shuffle=False라고 하면 된다. 그럼 1~60까지 순차적으로 나온다.
#shuffle=True는 섞여서 (기본 defalut값-입력안해도 저절로 나오는 것)
#print(x_train) 
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
#위에 실습일때 선생님이 val을 16개 할당하라고 하심
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,
                                                  train_size=0.8, shuffle=True)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

#2. 모델 구성
model=Sequential()
model.add(Dense(5, input_dim=1)) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(3))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#model.fit(x_train, y_train, epochs=100, validation_split=0.2)
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
#최적의 weight값 구하기 위해서는 x_train, y_train 입력해야함


#4. 평가예측
loss, mae=model.evaluate(x_test, y_test)
print('loss:', loss)
print('mae :', mae)

y_predict=model.predict(x_test) #즉, y_predict는 y_test와 비슷하게 되야함
print(y_predict)

#shuffle=True이면 false일때보다 mse낮다.
#loss: 0.020088428631424904
#mae : 0.11619345843791962

#validation_split=0.2이면
#loss: 0.5605039596557617
#mae : 0.610413670539856 따라서 신뢰도 낮다.

#validation_data=(x_val, y_val)
#loss: 0.0006479662843048573
#mae : 0.0214129276573658

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2) # 값이 1에 가까울 수록 좋다.
'''
