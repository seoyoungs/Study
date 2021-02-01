# m31로 만든 0.95 이상의 n_component =?를 사용해
# dnn 모델 만들어라
# mnist dnn 보다 성능 좋게
## 이때 전처리시 y는 따로 부여하지 않는다---> 그럼 훈련, 평가는 어떻게??

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

(x_train, _), (x_test, _) = mnist.load_data() # '_' -->y_train 과 y_test안하겠다는 것

x = np.append(x_train, x_test, axis = 0)
print(x.shape) #(70000, 28, 28)

#============================================== PCA로 컬럼 압축 
x = x.reshape(-1, 784)
# x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784) # 2차원으로 pca 하기위해

pca = PCA(n_components=155) # 파라미터 주성분 개수 
x2 = pca.fit_transform(x) #np.transform과 같음
# print(x2.shape) #(442, 8) 이렇게 컬럼 재구성

pca_EVR = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율
# print(pca_EVR) # 8개로 줄인 중요도에 대한 수치
print(sum(pca_EVR)) # 0.9504055742217271 pca에 대한 신뢰도

mnist = fetch_openml('mnist_784')
#=====전처리 keras==================================================
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, mnist.target,
                                     train_size = 0.8, random_state=44)

#다중분류 y원핫코딩

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y2_train)
y_test = to_categorical(y2_test)  
print(y2_test.shape, x2_test.shape) # (14000,) (14000, 155)
print(y2_train.shape, x2_train.shape) #  (56000,) (56000, 155)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model=Sequential()
model.add(Dense(units=12, activation='relu', input_shape=(155,)))
model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(units=1, activation='relu'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse',
              optimizer='sgd', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x2_train,y_train, epochs=10, 
           validation_split=0.2, batch_size=32,verbose=1)

#4. 평가 훈련
loss=model.evaluate(x2_test,y_test, batch_size=32)
print(loss)
y_pred = model.predict(x2_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

'''
Mnist
결과
[0.11182762682437897, 0.9659000039100647]

Dnn
loss: [0.18929392099380493, 0.9452999830245972]
y_pred:  [7 2 1 0 4 1 4 9 6 9]
y_test:  [7 2 1 0 4 1 4 9 5 9]

PCA, DNN (0.95)
[0.09999978542327881, 0.899996280670166]
y_pred:  [0 0 0 0 0 0 0 0 0 0]
y_test:  [8 4 6 2 6 1 0 7 9 7]
'''






