import numpy as np

x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

#print(x_data)
#print(y_data)
#print(x_data.shape, y_data.shape)

######모델을 완성 하시오

# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x_data,y_data, 
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
y_data= to_categorical(y_data)
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
                   #mean_squared_error
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=50, 
           validation_split=0.2, batch_size=10,verbose=1)

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print(loss)  #loss, accurac 값 추출
#y_pred=model.predict(x[-5:-1])
#print(y_pred) # y_pred로 코딩한 값
#print(y[-5:-1]) 

##argmax로 결과치 수정

y1_pred = model.predict(x_test[-5:-1])
print(y1_pred)
print(np.argmax(y1_pred, axis=1))


'''
[0.1219232827425003, 0.9333333373069763]
[[1.9889155e-12 5.6033628e-04 9.9943966e-01]
 [9.9923849e-01 7.6151703e-04 1.7252067e-14]
 [9.9588376e-01 4.1162698e-03 7.7915669e-12]
 [2.0276243e-06 6.0807681e-01 3.9192113e-01]]
[2 0 0 1]
'''
