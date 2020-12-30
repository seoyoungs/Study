#다:다 mlp
import numpy as np

#1.데이터
x=np.array([range(100),range(301, 401), range(1,101)])
y=np.array([range(711,811), range(1,101), range(201,301)])#각100개
#x,y는 dim=1이고 벡터이다. 스칼라 10개

print(x.shape)  #(3,100) 3행 100열
print(y.shape)  #(3,100) 

#x=np.array([1,2,3,4,5,6,7,8,9,10])
#결과(10,) x가 10개 스칼라라는 뜻
#x=np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
#결과 (2, 10) 2행 10열이라는 뜻 #input_dim=10인지 여쭤보기

x=np.transpose(x) #행렬 바꾸기1
#print(x.shape) #(100,3)로 바꿔준다. 100행 3열로 정리
#x=x.T #행렬 바꾸기2
#x=np.swapaxes(x,0,1) #행렬 바꾸기3

y=np.transpose(y) #행렬 바꾸기1
#print(y.shape)
from sklearn.model_selection import train_test_split
#우리 목적 train, test 분리(split)하는 것
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True,test_size=0.2, random_state=66)
#train_test_split 행을 정리하는 것이므로 transpose한 다음 하기
# shuffle=True 섞어주기
# random_state는 shuffle=True에서만 쓸 수 있다. 
# random_state=66, 랜덤 난수란 동일한 모델로 다른 훈련을 시킬 때
#동일한 조건에서 프로그램 돌리기 위해서 지정해 주는 것
print(x_train.shape) # (80, 3)
print(y_train.shape) #(80,3)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense #이렇게 해도 되는데 느리다

model=Sequential()
model.add(Dense(10, input_dim=3)) #행이 무시되고 열만 표시 3열
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(3))
# output_dim=3 y도 3차원 이므로

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train,batch_size=1, epochs=50,
          validation_split=0.2) #각 컬럼별 20%씩 쓰는 것

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
print("RMSE : ", RMSE(y_test,y_predict)) 
#MSE,mae,RMSE가 결과값으로 나온다. 이건 밑처럼 지정해주기
#print("mse: ",mean_squared_error(y_test, y_predict))
#print("mse: ",mean_squared_error(y_predict, y_test)) #위랑 살짝 값 다름

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2) # 값이 1에 가까울 수록 좋다.