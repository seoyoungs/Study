import numpy as np

#1.데이터
x=np.array([range(100),range(201,301), range(1,101), 
           range(100), range(301, 401)])
y=np.array([range(711,811), range(1,101)])#각100개
#x,y는 dim=1이고 벡터이다. 스칼라 10개

#print(x.shape)  #(5,100) 3행 100열
#print(y.shape)  #(2,100) 

#x=np.array([1,2,3,4,5,6,7,8,9,10])
#결과(10,) x가 10개 스칼라라는 뜻
#x=np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
#결과 (2, 10) 2행 10열이라는 뜻 #input_dim=10인지 여쭤보기

x=np.transpose(x) #행렬 바꾸기1
#print(x.shape) #(100,5)로 바꿔준다. 100행 3열로 정리
#x=x.T #행렬 바꾸기2
#x=np.swapaxes(x,0,1) #행렬 바꾸기3

y=np.transpose(y) #행렬 바꾸기1
#print(y.shape)  #(100,2)

x_pred2=np.array([100,402,101,100,401]) #행렬 바꾸기1
x_pred2=x_pred2.reshape(1,5) ###1차원인것도 차원변경해 행렬 바꿔줄수 있다.
#[[1,2,3,4,5]] 이런식으로 2차원 결과가 나온다. 그러면 input_dim=5이다.
#x의 dim이 5이므로 x_pred2도 5로 설정해야한다.
#print("x_pred2.shape : ", x_pred2.shape) #(1,5)로 나온다. 2차원으로 칼럼 5개다.

#x_pred2=np.transpose(x_pred2)
# 결과가 1차원이라 행렬 바꿔도 #(5,)로 똑같이 나온다.

from sklearn.model_selection import train_test_split
#우리 목적 train, test 분리(split)하는 것
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True,test_size=0.2, random_state=66)
#train_test_split 행을 정리하는 것이므로 transpose한 다음 하기
# shuffle=True 섞어주기
#random_state=66 난수생성기시드(뜻-랜덤 난수의 66번째꺼 사용한다는 뜻)
#print(x_train.shape) # (80, 5)
#print(y_train.shape) #(80,2)

#2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(5,))
aaa = Dense(5, activation='relu')(input1)
aaa=Dense(3)(aaa)
aaa=Dense(4)(aaa)
outputs=Dense(2)(aaa)
#차례대로 앞에 있던게 맨 뒤로 온다. input을 뒤에 명시
model=Model(inputs=input1, outputs=outputs)
model.summary()
#인풋과 아웃풋 dim 맞추기
'''
model=Sequential()
#model.add(Dense(10, input_dim=5)) #행이 무시되고 열만 표시 3열
model.add(Dense(5,activation='relu', input_shape=(5,))) 
#input_shape=(5,) 컬럼이 5개라는 말
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.summary()
# output_dim=2 y도 3차원 이므로 Dense(2)
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train,batch_size=1, epochs=50,
          validation_split=0.2, verbose=0) #각 컬럼별 20%씩 쓰는 것
#4. 평가예측
loss, mae =model.evaluate(x_test,y_test) #훈련하는 것이므로 test로 할당하기
print('loss : ', loss)
print('mae : ', mae)
y_predict=model.predict(x_test) 
#x로 하면shape가 (100,3)이므로 모양틀리다. y_test와 맞춰야한다. 
#print(y_predict)
#사이킷런
from sklearn.metrics import mean_squared_error
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임
print("RMSE : ", RMSE(y_test,y_predict)) #MSE,mae,RMSE가 결과값으로 나온다.
#print("mse: ",mean_squared_error(y_test, y_predict))
#print("mse: ",mean_squared_error(y_predict, y_test))
#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2) # 값이 1에 가까울 수록 좋다.
#다만들면 predict의 일부값을 출력하시오
y_pred=model.predict(x_pred2)
print(y_pred)
'''