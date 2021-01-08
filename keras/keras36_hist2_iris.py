#hist를 이용해 그래프를 그려보시오
#loss, val_loss, acc, val_acc
###sklearn 다중분류 모델 (acc적용)
#accuracy확인

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

#### 원핫인코딩 oneHotEcoding####
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(150,3)
#print(y)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
#model.add(Dense(1)) #-->hidden이 없는 layer가 있다. 
# activation 다음 층으로 연결할때 전달하는 방법
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)
#분류하고자 하는 노드에 개수를 output나오게 해라
#linear은 선형, relu는 회귀, sigmoid는 이진분류

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#얼리 스탑핑 적용
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='loss', patience=10, mode='auto')
hist = model.fit(x,y, epochs=500, batch_size=16, verbose=1, 
         validation_split=0.2, callbacks=[es])

print(hist)
print(hist.history.keys())
#print(hist.history['loss']) ###로스값이 차례대로 줄어드는 것을 볼 수 있다.
##그림을 loss값을 토대로 그릴 예정
####그래프####
import matplotlib.pyplot as plt
#만약 plt.plot(x,y)하면 x,y 값이 찍힌다.
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss&acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()
