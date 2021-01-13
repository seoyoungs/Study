
import numpy as np

x= np.load('../data/npy/boston_x.npy')
y = np.load('../data/npy/boston_y.npy')

x=x.reshape(x.shape[0], x.shape[1],1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.7, random_state=104)
from sklearn.preprocessing import MinMaxScaler

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv1D, Flatten

model=Sequential()
model.add(Conv1D(filters=15, kernel_size=1, padding='same',
                   strides=1 , input_shape=(13,1))) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(4, activation='relu'))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(1))


#컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath2 = '../data/modelCheckpoint/k50_2_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=10)
cp =ModelCheckpoint(filepath=modelpath2, monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='mae',
              optimizer='adam', metrics=['mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=100, batch_size=16, verbose=1,
                validation_split=0.2,callbacks=[es, cp])

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test, batch_size=10)
print('loss, mae : ', loss, mae)
y_predict= model.predict(x_test)
'''
#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ',RMSE(y_test,y_predict))

#R2
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('R2 : ', r2)
'''
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

'''
#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict)) 

#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''
'''
 LSTM  전
결과
loss: 19.869380950927734
mae : 3.2330851554870605
RMSE :  4.457507891337804
R2 :  0.7622795125940092

LSTM 후
loss: 15.670780181884766
mae : 2.846346378326416
RMSE :  3.958633703813784
R2 :  0.8125122028343794

Conv1d
loss: 85.08113098144531
mae : 6.62658166885376
'''
