### 인공지능계의 hello wkrld라고 불리는 mnist
### 다중분류이다.(60,000개의 트레이닝 데이터와 10,000개)
##https://buomsoo-kim.github.io/keras/2018/05/05/Easy-deep-learning-with-Keras-11.md/


#####실습 모델을 완성하시오
#자료는 acc (model.fit에 metrics값) /0.985 이상 나오게 해라
#응용
#y_test 10개와 y_pred 10개를 출력하시오
#y_test[:10]: =(??????)
#y_pred[:10]: =(??????)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#print('x_train[0]: ', x_train[0])
#print('y_train[0]: ', y_train[0]) # 5
#print(x_train[0].shape) #(28, 28)

#plt.imshow(x_train[0])
#plt.imshow(x_train[0], 'gray') #이렇게 gray를 해야 제대로 됨
#plt.show()
#그림에서 특성이 없는는 것은 검은색
# 특성이 제일 밝은 것은 255이다. -> 흰색일수록 특성 있음

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
model.add(Conv2D(filters=50, kernel_size=(3,3), 
                padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(5))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
#model.add(Conv2D(???))
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=20, 
           validation_split=0.2, batch_size=16,verbose=1)

#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=16)
print(loss)

y_pred = model.predict(x_test)
#R2 (accuracy대산 R2를 사용한다.)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

'''
Mnist
결과
[0.11182762682437897, 0.9659000039100647]
'''





