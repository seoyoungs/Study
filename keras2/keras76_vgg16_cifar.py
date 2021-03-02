# 실습
# cifar10으로 vgg16 넣어서 만들것

# 결과치에 대한 기존값과 비교
# 전이학습은 preprocess_input이라는 전처리, True로 할꺼면 안해도 됨

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ============== 전처리 ===================
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

#다중분류 y원핫코딩
y_train = to_categorical(y_train) #(50000, 10)
y_test = to_categorical(y_test)  #(10000, 10)

# ============== 모델링 =====================
vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(32, 32, 3))
# print(model.weights)

vgg16.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
vgg16.summary()
# 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
print(len(vgg16.weights)) # 26
print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16) # 3차원 -> layer 26개
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(10, activation='softmax'))
model.summary()

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
vgg16
Loss :  1.1038784980773926
acc :  0.6144999861717224
'''
