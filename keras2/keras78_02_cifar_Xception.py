# 실습
# cifar10으로 vgg16 넣어서 만들것

# 결과치에 대한 기존값과 비교
# 전이학습은 preprocess_input이라는 전처리, True로 할꺼면 안해도 됨

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.models import Model, Input

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

Xception = Xception(weights = 'imagenet', include_top=False, input_shape=(96, 96, 3))
# print(model.weights)

# ============== 전처리 ===================
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_train = x_train.astype('float32')/255.  # 전처리
x_test = x_test.astype('float32')/255.  # 전처리

#다중분류 y원핫코딩
y_train = to_categorical(y_train) #(50000, 10)
y_test = to_categorical(y_test)  #(10000, 10)

# ============== 모델링 =====================
Xception.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
Xception.summary()
# 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
print(len(Xception.weights)) # 26
print(len(Xception.trainable_weights)) # 0

model = Sequential()
model.add(UpSampling2D(size=(3,3)))
model.add(Xception)
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=20, 
           validation_split=0.2, batch_size=16,verbose=1)

#4. 평가 훈련
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("Loss : ", loss)
print("acc : ", acc)


'''
model = Sequential()
model.add(UpSampling2D(size=(3,3)))
model.add(xception)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['accuracy'])
model.fit(x_train,y_train,validation_split=0.2,epochs=1000,batch_size=128,callbacks=[es,lr])

upsampling을 이용
XCEPTION을 넣으면 32,32,3이 upsampling이 3일 때를 고려
(32,32,3)인것을 -> (96,96,3) 해상도를 늘리는 함수 upsampling(3,3) 이용
업샘플링 레이어(첫번째에 있음)을 지날 떄 나의 데이터의 크기(32,32,3)이 3배로 늘어나니까 
// 전이학습 정의할 떄 크기를 바로 (96,96,3) 으로 지정한다
'''
'''
vgg16
Loss :  1.1038784980773926
acc :  0.6144999861717224

vgg19
Loss :  1.115734577178955
acc :  0.6082000136375427

xception
Loss :  2.302741289138794
acc :  0.10000000149011612
upsampling을 이용
XCEPTION을 넣으면 32,32,3이 upsampling이 3일 때를 고려
(32,32,3)인것을 -> (96,96,3) 해상도를 늘리는 함수 upsampling(3,3) 이용
업샘플링 레이어(첫번째에 있음)을 지날 떄 나의 데이터의 크기(32,32,3)이 3배로 늘어나니까 
// 전이학습 정의할 떄 크기를 바로 (96,96,3) 으로 지정한다


'''