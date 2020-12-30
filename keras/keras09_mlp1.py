#다:1 mlp
import numpy as np
x=np.array([[1,2,3,4,5,6,7,8,9,10],
           [11,12,13,14,15,16,17,18,19,20]])
y=np.array([1,2,3,4,5,6,7,8,9,10])
#x,y는 dim=1이고 벡터이다. 스칼라 10개

print(x.shape) 
#x=np.array([1,2,3,4,5,6,7,8,9,10])
#결과(10,) x가 10개 스칼라라는 뜻
#x=np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
#결과 (2, 10) 2행 10열이라는 뜻 #input_dim=10인지 여쭤보기

x=np.transpose(x) #행렬 바꾸기1
print(x.shape) #(10,2)로 바꿔준다. 10행 2열로 정리
#x3=x.T #행렬 바꾸기2
#x4=np.swapaxes(x,0,1) #행렬 바꾸기3

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense #이렇게 해도 되는데 느리다

model=Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y,batch_size=1, validation_split=0.2) #각 컬럼별 20%씩 쓰는 것

#4. 평가예측
loss, mae =model.evaluate(x,y)
print('loss : ', loss)
print('mae : ', mae)

y_predict=model.predict(x)
#print(y_predict)
'''
#사이킷런
from sklearn.metrics import mean_squared_error
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임
print("RMSE : ", RMSE(y_test,y_predict)) #MSE,mae RMSE가 결과값으로 나온다.
#print("mse: ",mean_squared_error(y_test, y_predict))
print("mse: ",mean_squared_error(y_predict, y_test))

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2) # 값이 1에 가까울 수록 좋다.
'''