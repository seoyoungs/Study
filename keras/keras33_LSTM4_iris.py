#사이킷
#LSTM으로 모델링
#DENASE와 성능비교
#다중분류

#accuracy확인

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
print(y)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2] --> 이렇게 0,1,2 섞어서 나온다. 0,1만 나오게 수정해야함
'''

######sklearn로 전처리 하기
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
e = LabelEncoder()
e.fit(y)
y1 = e.transform(y)
y1 = np_utils.to_categorical(y1)
print(y1.shape)

# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test= train_test_split(x,y1, 
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

'''
#### 원핫인코딩 oneHotEcoding####
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(150,3)
#print(y)#벡터형식으로 바뀐다. 0,1,2 사라지고 벡터로 변경
[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
'''

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)
#분류하고자 하는 노드에 개수를 output나오게 해라
#linear은 선형, relu는 회귀, sigmoid는 이진분류

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc', 'mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y1_train, epochs=150, 
           validation_split=0.2, batch_size=10,verbose=1)

#4. 평가 ,예측
loss=model.evaluate(x_test,y1_test, batch_size=10)
print(loss)  #loss, accurac 값 추출
#y_pred=model.predict(x[-5:-1])
#print(y_pred) # y_pred로 코딩한 값
#print(y[-5:-1]) 

'''
y_pred=model.predict(x[-5:-1])
print(y_pred) -원핫인코딩 한값
[2.3917291e-23 3.4554517e-09 1.0000000e+00]
 [2.9042364e-21 4.1744691e-08 1.0000000e+00]
 [5.4568571e-22 2.7116839e-08 1.0000000e+00]
 [1.2453324e-22 1.7178943e-08 1.0000000e+00]]
원래 y값
[[0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
'''

##argmax로 결과치 수정

y1_pred = model.predict(x_test[-5:-1])
print(y1_pred)
print(np.argmax(y1_pred, axis=1))


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
'''



#####keras33 5개는 원래 LSTM이 낮게 나오는게 맞다
####만약 아니면 Dense를 잘못한것
