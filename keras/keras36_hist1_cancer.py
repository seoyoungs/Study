#hist를 이용해 그래프를 그려보시오
#loss, val_loss, acc, val_acc
###sklearn 이진분류 모델 (acc적용)
# 이진 분류 예제
# 실습
#acc = 0.98이상, 
#y_predict 값도 추출 변경
#####과제 if구문으로 데이터셋 0~1로 만지기

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

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
#model.add(Dense(1)) #-->hidden이 없는 layer가 있다.
model.add(Dense(10, activation='relu')) 
# activation 다음 층으로 연결할때 전달하는 방법
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #마지막에만 sigmoid를 준다
#input, output에 sigmoid를 줘도 된다. 각자의 판단을 함수 적용

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#얼리 스탑핑 적용
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = model.fit(x,y, epochs=500, batch_size=16, verbose=1, 
         validation_split=0.2, callbacks=[es])

print(hist)
print(hist.history.keys())
#print(hist.history['loss']) ###로스값이 차례대로 줄어드는 것을 볼 수 있다.
##그림을 loss값을 토대로 그릴 예정
####그래프####
import matplotlib.pyplot as plt
#만약 plt.plot(x,y)하면 x,y 값이 찍힌다.
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss&acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()