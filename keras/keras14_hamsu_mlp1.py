#다:1 함수형
#keras10_mlp2.py를 함수형으로

#다:1 mlp
#실습 train과 test 분리해 소스를 완성하세요.
#(test와 train할당하는 지점 잘 보기)
#선생님이 준 데이터는 열이 중심이다. 근데 split은 행 중심이니 바꾸기

#1.데이터
import numpy as np
x=np.array([range(100), range(301,401), range(1,101)])
y=np.array(range(711,811))

#print(x.shape) #(3,100) 3행 100열 
#print(y.shape) #(100,) 스칼라가 100개 벡터는 1

#x=np.array([1,2,3,4,5,6,7,8,9,10])
#결과(10,) x가 10개 스칼라라는 뜻
#x=np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
#결과 (2, 10) 2행 10열이라는 뜻 #input_dim=10인지 여쭤보기

x=np.transpose(x)
#print(x.shape) #(100, 3) 100행 3열로 사이킷런에 train_test_split 적용 가능
#x=x.T #행렬 바꾸기2
#x=np.swapaxes(x,0,1) #행렬 바꾸기3
y=np.transpose(y)
#print(y.shape)

#train, test 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                   shuffle=True, test_size=0.2, random_state=66)
#train_test_split 행을 정리하는 것이므로 transpose한 다음 하기
# shuffle=True 섞어주기
# random_state=66
print(x_train.shape) # (80, 3)
print(y_train.shape)

#print(x_train) 
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input

input1=Input(shape=(3,))
aaa = Dense(15, activation='relu')(input1)
aaa=Dense(13)(aaa)
aaa=Dense(7)(aaa)
aaa=Dense(3)(aaa)
outputs=Dense(1)(aaa)
#차례대로 앞에 있던게 맨 뒤로 온다. input을 뒤에 명시
model=Model(inputs=input1, outputs=outputs)
model.summary()
#인풋과 아웃풋 dim 맞추기
'''
model=Sequential()
model.add(Dense(15, input_dim=3)) #transpose한 x열 3열 이므로
model.add(Dense(13))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))
'''
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, 
           epochs=100, validation_split=0.2) #각 컬럼별 20%씩 쓰는 것

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test) #훈련하는 것이므로 test로 할당하기
print('loss : ', loss) 
print('mae : ', mae)
y_predict=model.predict(x_test)
#x로 하면shape가 (100,3)이므로 모양틀리다. y_test와 맞춰야한다. 
#print(y_predict)

#사이킷런 RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 뜻 RMSE라는 함수 정의 하겠다는 뜻 
#파이썬을 줄바꿈으로 점위를 표시한다. return줄 바꿈한 것보기
#mean_squared_error와 mse가 비슷한 의미로 쓰임
print("RMSE : ", RMSE(y_test, y_predict)) #MSE,mae,RMSE가 결과값으로 나온다.
#print("mse: ",mean_squared_error(y_test, y_predict))
#print("mse: ",mean_squared_error(y_predict, y_test))

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print("R2 :", r2) # 값이 1에 가까울 수록 좋다.

'''
결과
loss :  1.3224780381904111e-08
mae :  0.0001068115234375
RMSE :  0.00012283088255308903
R2 : 0.9999999999809099
'''