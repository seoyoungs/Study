##실습 
# dropout 적용
##다중 분류 acc사용

import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris() #x,y=load_iris(return_X_y=True)와 같다
x= dataset.data
y= dataset.target
#print(dataset.DESCR)
#print(dataset.feature_names)

#print(x.shape) #(150,4)
#print(y.shape) #y가 3종류(150,)---> 바뀔거다
#print(x[:5])
'''
from sklearn.preprocessing import OneHotEncoder 
#sklearn 데이터 다중분류 경우 이거 사용
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1,1)).toarray()
'''
# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
#print(x_train.shape) #(120,4)
print(x_test.shape) #(30,4)
x_train= x_train.reshape(120, 4,1,1)
x_test= x_test.reshape(30, 4,1,1)

#### 원핫인코딩 oneHotEcoding####  ----> 이거 텐져플로우 데이터인 경우
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(150,3)
#print(y)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1), 
                padding='same', strides=(1,1), input_shape=(4,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Dense(4, activation='relu'))
model.add(Flatten())
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(3, activation='softmax')) #마지막에만 sigmoid를 준다

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath7 = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=5)
cp =ModelCheckpoint(filepath='modelpath7', monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1,
                 validation_split=0.2,callbacks=[es, cp])

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print('loss:', loss)  #loss, accurac 값 추출
y_pred=model.predict(x_test[-5:-1])
print(np.argmax(y_pred, axis=1))
print(y_pred) # y_pred로 코딩한 값
print(y_test[-5:-1]) #원래 기존 y값

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

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

#plt.title('정확도')
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

#plt.title('정확도')
plt.title('accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()
