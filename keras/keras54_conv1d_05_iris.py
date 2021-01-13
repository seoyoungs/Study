import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris() #x,y=load_iris(return_X_y=True)와 같다
x= dataset.data
y= dataset.target

# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#print(x_train.shape) #((120, 4))
#print(x_test.shape) #(30, 4)
x_train=x_train.reshape(120,4,1)
x_test=x_test.reshape(30,4,1)

#### 원핫인코딩 oneHotEcoding####  ----> 이거 텐져플로우 데이터인 경우
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(150,3)
#print(y)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=15, kernel_size=1, padding='same',
                   strides=1 , input_shape=(4,1)))
model.add(Dense(5, activation='relu'))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)
#분류하고자 하는 노드에 개수를 output나오게 해라
#linear은 선형, relu는 회귀, sigmoid는 이진분류

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc', 'mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=150, 
           validation_split=0.2, batch_size=10,verbose=1)

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print(loss)  #loss, accurac 값 추출
#y_pred=model.predict(x[-5:-1])
#print(y_pred) # y_pred로 코딩한 값
#print(y[-5:-1]) 

##argmax로 결과치 수정

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(np.argmax(y_pred, axis=1))


'''
LSTM 전 결과
[0.2929254472255707, 0.9666666388511658, 0.15748263895511627]
[[1.4803344e-03 8.8146053e-02 9.1037357e-01]
 [9.9200785e-01 7.4190465e-03 5.7309697e-04]
 [7.6894391e-01 2.2478288e-01 6.2731630e-03]
 [1.1676942e-02 5.5393237e-01 4.3439060e-01]]
[2 0 0 1]


LSTM 후 결과
[0.1451122760772705, 0.9666666388511658, 0.07901684194803238]
[[1.22167054e-14 2.29335134e-03 9.97706652e-01]
 [9.94968832e-01 5.03108837e-03 8.31697307e-08]
 [9.89029050e-01 1.09656295e-02 5.31375554e-06]
 [2.45542378e-06 5.43140948e-01 4.56856608e-01]]
[2 0 0 1]

conv1d
 loss: 0.4191 - acc: 0.8333 - mae: 0.1979
[0.41905319690704346, 0.8333333134651184, 0.19789481163024902]
'''