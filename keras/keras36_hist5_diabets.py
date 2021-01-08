#hist를 이용해 그래프를 그려보시오
#loss, val_loss
#### 회귀모델 acc적용 x


import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

#print(x[:5])
#print(y[:10])
#print(x.shape, y.shape) #(442, 10) - 10열개(10,0) (442,) -output1
#print(np.max(x), np.min(x))
#print(dataset.feature_names)
#print(dataset.DESCR)

#x = x/0.198787989657293
#데이터 전처리 libarary
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) #---> x_train으로 변경
x=scaler.transform(x)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                           shuffle=True, train_size=0.8, random_state=66)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

inputs1=Input(shape=(10,))
aaa= Dense(15, activation='relu')(inputs1)
aaa= Dense(8, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
aaa= Dense(3, activation='relu')(aaa)
aaa= Dense(5, activation='relu')(aaa)
outputs=Dense(1)(aaa)
model= Model(inputs= inputs1, outputs=outputs)
#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

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

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'])
plt.show()
