import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# model =Sequential()
# model.add(Conv2D(filters=50, kernel_size=(3,3), 
#                 padding='same', strides=(1,1), input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dense(5))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()

#model.save('../data/h5/k52_1_model1.h5')  #2. 모델링 까지 저장(가중치 안들어가있음)

#3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelCheckpoint/k52_2_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
#es= EarlyStopping(monitor='val_loss', patience=5)
#cp =ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    #save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['acc'])
# ####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
# model.load_weights('../data/h5/k52_1_weight.h5') #이미 모델은 save된 상태에서 weight하기
#이미 fit까지 포함되서 weight값 처리,,,,,,근데 모델은 저장안됨 fit만 저장.(모델, 컴파일은 필요)

'''
hist = model.fit(x_train,y_train, epochs=10, batch_size=8, verbose=1,
                 validation_split=0.2,callbacks=[es, cp]) #wieght 생성 지점
model.save('../data/h5/k52_1_model2.h5') #fit까지
#총 모델 save두번됨 원하는 장소에 저장 넣음
model.save_weights('../data/h5/k52_1_weight.h5')
from tensorflow.keras.models import Sequential, load_model
model1 = load_model('../data/h5/k52_1_model2.h5')
'''

#4_1. 평가, 예측
# result=model.evaluate(x_test,y_test, batch_size=8)
# print('가중치_loss : ', result[0])
# print('가중치_acc : ', result[1])  #모델 3개 다 출력

# #model.load_weights('../data/h5/k52_1_weight.h5') #근데 모델은 저장안됨 fit만 저장
# model2 = load_model('../data/h5/k52_1_model2.h5') #가중치와 모델이 포함
# result2=model2.evaluate(x_test,y_test, batch_size=8)
# print('로그모델_loss : ', result2[0])
# print('로그모델_acc : ', result2[1]) 

model2 = load_model('../data/modelCheckpoint/k52_mnist_checkpoint.hdf5') #modelCheckpoint 가중치와 모델이 포함?
result2=model2.evaluate(x_test,y_test, batch_size=8)
print('로그체크포인트_loss : ', result2[0])
print('로그체크포인트_acc : ', result2[1])

'''
model2 = load_model('../data/modelCheckpoint/k52_mnist_checkpoint.hdf5')
이제 이 값으로 동일하게 나온다. wight 값이 정해졌으므로 (모델, 훈련 모두 저장---model.save기능과 같다.MpdelCheckPoint)
로그체크포인트_loss :  0.059340156614780426
로그체크포인트_acc :  0.9821000099182129
model1 = load_model('../data/h5/k52_1_model2.h5') #두자리에서 지정가능(모델 밑, 훈련(fit)밑)
로그모델_loss :  0.06455615162849426
로그모델_acc :  0.9782000184059143
load_weight의 경우는 모델이랑 컴파일은 언급을 해야한다.
이 두개 값 중에 check포인트가 값이 더 좋다.
왜? early는 훈련중 원하는 낮게 나오는 지점이 반복되면 멈추고
체크포인트는 젤 낮은 지점에서 낮은 지점을 하는게 좋다.
'''
