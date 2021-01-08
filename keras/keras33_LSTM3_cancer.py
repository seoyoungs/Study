#사이킷
#LSTM으로 모델링
#DENASE와 성능비교
#이중분류
#분류에는 R2쓰는 거 아니다.


'''
y[-5:-1] =?
y_pred=model.predict(x[-5:-1]) -> 끝부터 끝에서 5번째구해라
-1은 가장 끝값, 0은 시작값
print(y_pred)
print(y[-5:-1])
왜 predict 값 0,1로 나오는 거 아닌지 알아보기
'''
import numpy as np
from sklearn.datasets import load_breast_cancer

#1.데이터
datasets = load_breast_cancer()

#print(datasets.DESCR)
#print(datasets.feature_names)

x= datasets.data
y= datasets.target
#print(x.shape) #(569,30) # 실질적 칼럼개수(32개, id와 y칼럼빼고)
#print(y.shape) #(569,)
#y값은 diagnosis 진단 여부(B(양성), M(악성))
#print(x[:5])
#print(y)

# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#print(x_train.shape) #(455, 30)
#print(x_test.shape) #(114, 30)
x_train=x_train.reshape(455,30,1)
x_test=x_test.reshape(114,30,1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(30,1)))
#model.add(Dense(1)) #-->hidden이 없는 layer가 있다.
model.add(Dense(10, activation='relu')) 
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #마지막에만 sigmoid를 준다
#input, output에 sigmoid를 줘도 된다. 각자의 판단을 함수 적용

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['acc', 'mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=50,  batch_size=10,
validation_split=0.2, verbose=1)

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print(loss)  #loss, accurac 값 추출
y_pred=model.predict(x_test[-5:-1])
print(np.argmax(y_pred, axis=1))
print(y_pred) # y_pred로 코딩한 값
print(y_test[-5:-1]) #원래 기존 y값
#y_predict=model.predict(x_test)
#print(y_predict) #왜 소수점이야?

#######분류에는 R2와 같은 accuracy_score 있다.


'''
값
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(30,)))
#model.add(Dense(1)) #-->hidden이 없는 layer가 있다.
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
[0.30694854259490967, 0.9033392071723938]
[0.2665921747684479, 0.9332161545753479]
#전처리 했을 때
[0.11432956904172897, 0.9824561476707458]
print(y_pred) #왜 소수점이야?
print(y[-5:-1]) 
답
[0 0 0 0]
[[8.8095903e-01]
 [8.8095903e-01]
 [8.8095903e-01]  #왜 소수점이야?
LSTM한후
[0.22490595281124115, 0.9122806787490845, 0.12303196638822556]
[0 0 0 0]
[[0.9996244 ]
 [0.06398312]
 [0.9900815 ]
 [0.9885916 ]]
[1 0 1 1]
'''
#if문으로 0과 1사이로 변환하기print(y_pred)값
