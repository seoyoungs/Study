# keras_18_6복사
##머신러닝 데이터는 sklearn에서 많이 제공한다.
###### Minmaxscaler 데이터 전처리를 통해 0~1까지로 만들기
#### standardscaler 데이터 한쪽으로 몰렸을때 사용(골고루 퍼뜨림)

import numpy as np #gpu단순연산에 좋다.
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_boston
#사이킷런에서 데이터 기본 제공

#1. 데이터
dataset= load_boston()
x=dataset.data
y=dataset.target
#print(x.shape) #(506, 13)
#print(y.shape)  #(506,)
#print("==============")
#print(x[:5])
#print(y[:10])

'''
전처리 방법 1(0~1사이로 전처리)
print(x[:5])
[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00
  6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02
  4.9800e+00] ----> 이런게 5개씩 13열---> 이렇게 데이터 전처리를 해야한다.
 -- 0~1사이에 있다.
print(y[:10])
[24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9]---> 값 다양(회귀모델)
'''

print(np.max(x), np.min(x)) #711.0 0.0(이렇게 1보다 클수 있다. 전처리방법 다양) 
#이걸보면 전처리 전 데이터인 것을 알 수 있다.(0~1사이가 아니기 때문)
#print(dataset.feature_names) #x의 데이터 이름들
#print(dataset.DESCR) #데이터 셋 이름과 설명
'''
#####데이터 전처리 (minMax)
x = x/711. #왜 711. 하는 이유---타입 때문에(실수형)
근데 이거 틀리다. (x-열)컬럼마다 최소값 틀리기 때문
그냥 쓰면 711 정수형이 된다.
 X=(X - 최소) / (최대 - 최소)
  =(x - np.min(x)) / (np.max(x) - np.min(x))
'''
# -------------------데이터 전처리 libarary -----------------------------
# https://m-insideout.tistory.com/5 PowerTransformer에 관해서
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer

scaler = MinMaxScaler(feature_range=(1, 2))
power = PowerTransformer(method='box-cox') #log화해 정규분포화 하는 것
pipeline = Pipeline(steps=[('s', scaler),('p', power)])
x= pipeline.fit_transform(x)

# scaler = RobustScaler()
# scaler = StandardScaler()
# scaler = QuantileTransformer() # default : 균등분포
# scaler = QuantileTransformer(output_distribution='normal') #정규분포
# scaler = PowerTransformer()# 디폴트 method='yeo-johnson'
# scaler = PowerTransformer(method= 'Box-Cox')
# # scaler = MaxAbsScaler()
# scaler.fit(x) # ---> x_train으로 변경
# x=scaler.transform(x)
'''
#minmax ---2차원에서만 가능
print(np.max(x), np.min(x)) #711.0 0.0 -> 1.0 0.0
print(np.max(x[0])) #0.99999
#근데 이것도 문제가 있다.--대체 why....?
# StandardScaler --> 2차원
print(np.max(x), np.min(x)) #711.0 0.0 -> 0.44105193260704206
print(np.max(x[0])) # -> 0.44105193260704206
'''

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

model=Sequential()
model.add(Dense(12, activation='relu', input_dim=13)) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(4, activation='relu'))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, verbose=1)

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test,batch_size=1)
print('loss:', loss)
print('mae :', mae)
#print('loss, mae :', loss, mae)
y_predict=model.predict(x_test) 

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict)) 

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2:', r2)

'''
전처리 전
결과
loss: 19.869380950927734
mae : 3.2330851554870605
RMSE :  4.457507891337804
R2 :  0.7622795125940092
전처리 후
loss: 11.747001647949219
mae : 2.6047136783599854
RMSE :  3.427389877968649
R2 :  0.859456944795199
이렇게 전처리 하면 성능이 아주조금 향상된다.
반드시 전처리 해야한다.
MinMaxScalar전처리 결과 향상
x통째로 전처리
결과
loss: 11.236867904663086
mae : 2.424912452697754
RMSE :  3.352143695825399
R2 :  0.8655602720342852
StandardScaler 전처리 후
loss: 10.651695251464844
mae : 2.2952589988708496
RMSE :  3.2636933353064923
r2: 0.8725613800261594
# ---------------- outlier 처리에 효과적 ------------
RobustScaler
loss: 9.253771781921387
mae : 2.1508092880249023
RMSE :  3.0420014718944133
r2: 0.8892863394083234
QuantileTransformer -> 분위수자체를 1000개의 quantile(데이터가 1000개 이하면 효과 별로)
loss: 15.428872108459473
mae : 2.4710402488708496
RMSE :  3.927960555681241
r2: 0.8154064140418951
'''