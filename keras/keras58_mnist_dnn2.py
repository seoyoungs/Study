import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#다중분류이므로 원핫코딩, 전처리 해야한다.
##이걸로 전처리 해서 MinMaxscaler안해도 된다.
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.
#이미지 특성맞춰 숫자 바꾸기 x의 최대가 255이므로 255로 나눈다.
#이렇게 하면 0~1 사이로 된다.
#x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
#x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model =Sequential()
model.add(Dense(64,input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(5))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Flatten()) #만약 4차원으로 출력되면 이것도 괜찮다.
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#ReduceLROnPlateau ---> learning late감소하겠다.
modelpath = '../data/modelCheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
es= EarlyStopping(monitor='val_loss', patience=6)
cp =ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factor = 0.5, verbose=1)
#ReduceLROnPlateau----3번까지 개선안되는 거 되는데 그 이후로는 0.5로 깍겠다.
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1,
                 validation_split=0.5,callbacks=[es, cp, reduce_lr])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

#ReduceLROnPlateau
# Epoch 00017: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# optimizer='adam'일 때 0.0005로 0.5 깍여서 들어간것 이므로 원래 adam 디폴트 값은 0.001이다.
# 이렇게 안되면 0.5로 드랍한다고 나온다.

####시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.figure(figsize=(10,6)) #가로세로
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

plt.title('손실비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue') #여기에 lable안하고plt.legend에 라벨 입력
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy']) #직접 라벨명 입력
plt.show()
