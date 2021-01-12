import numpy as np

x=np.load('../data/npy/boston_x.npy')
y=np.load('../data/npy/boston_y.npy')

x=x.reshape(x.shape[0], x.shape[1],1,1) #=x.reshape(-1,13,1,1)
print(x.shape) #((506, 13, 1, 1))

#데이터 처리 (train, test분리 --필수!!!)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, shuffle=True, 
                                           train_size=0.8, random_state=66) #랜덤지정

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model=Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1), 
                padding='same', strides=(1,1), input_shape=(13,1,1)))
model.add(Dense(25, activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath1 = '../data/modelCheckpoint/k50_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=10)
cp =ModelCheckpoint(filepath=modelpath1, monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='mae',
              optimizer='adam', metrics=['mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=10, batch_size=16, verbose=1,
                validation_split=0.2,callbacks=[es, cp])

#4. 평가예측
loss=model.evaluate(x_test, y_test,batch_size=16)
print('loss:', loss)
y_pred = model.predict(x_test)
#R2 (accuracy대산 R2를 사용한다.)
'''
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)
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

