#hist를 이용해 그래프를 그려보시오
#loss, val_loss, acc, val_acc
###sklearn 다중분류 모델 (acc적용)


from sklearn.datasets import load_wine #sklearn데이터 전처리onehot으로

dataset = load_wine()
#print(dataset.DESCR)  #열 13개 y 3개
#print(dataset.feature_names)

x=dataset.data
y=dataset.target

#######sklearn 중요######
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1,1)).toarray()


###데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(13,)))
'''
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(8)) 
model.add(Dense(5)) 
'''
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)

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
