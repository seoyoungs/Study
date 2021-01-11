import numpy as np #gpu단순연산에 좋다.

from sklearn.datasets import load_boston
#사이킷런에서 데이터 기본 제공

#1. 데이터
dataset= load_boston()
x=dataset.data
y=dataset.target

#print(x.shape) #(506. 13)
#print(y.shape) #(506)

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
'''
model=Sequential()
model.add(Dense(64, activation='relu', input_dim=13)) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(40, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))
'''


model = Sequential([
    Dense(128,activation = 'relu',input_dim = 13),
    # model.add(Dense(10, activation='relu',input_shape=(13,))
    Dense(128),
    Dense(64),
    Dense(64),
    Dense(32),
    Dense(32),
    Dense(16),
    Dense(16),
    Dense(1),
])

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath4 = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=15, mode='auto')
cp =ModelCheckpoint(filepath='modelpath4', monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='mse',
              optimizer='adam', metrics=['mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=150, batch_size=8, verbose=1,
                validation_split=0.2,callbacks=[es, cp])

#4. 평가예측
loss=model.evaluate(x_test, y_test,batch_size=8)
print('loss:', loss)
#R2 (accuracy대산 R2를 사용한다.)
y_predict=model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

####시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) #가로세로 #이거 한 번 만 쓰기
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

#plt.title('손실비용') #한글깨짐 오류 해결할 것 과제1
plt.title('Cost loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()
