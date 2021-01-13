import numpy as np

x_train = np.load('../data/npy/fashion_x_train.npy')
y_train = np.load('../data/npy//fashion_y_train.npy')
x_test= np.load('../data/npy/fashion_x_test.npy')
y_test =np.load('../data/npy/fashion_y_test.npy')

x_train = x_train.reshape((x_train.shape[0], -1,1))/255.
x_test = x_test.reshape((x_test.shape[0], -1,1))/255.
print(x_train.shape)
'''
# 학습 과정을 단축시키기 위해 학습 데이터의 1/3만 활용한다
from sklearn.model_selection import train_test_split
x_train, _ , y_train, _ = train_test_split(x_train, y_train, 
                          test_size = 0.67, random_state = 7)
'''
#다중분류 y원핫코딩

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
model=Sequential()
model.add(Dense(units=12, activation='relu', input_shape=(28*28,1)))
model.add(Dense(20,activation='relu'))
model.add(Flatten())
model.add(Dense(15,activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#3. compile
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=15, 
           validation_split=0.2, batch_size=32,verbose=1)

#4. 평가 훈련
loss=model.evaluate(x_test,y_test, batch_size=32)
print(loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))


'''
conv1d
[0.38242435455322266, 0.8669999837875366]
'''