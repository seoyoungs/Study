import numpy as np
import pandas as pd

x = np.load('../data/npy/sam2.npy',allow_pickle=True)[0]
y = np.load('../data/npy/sam2.npy',allow_pickle=True)[1]

# x_pred = x_pred.reshape(-1,5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(2), padding='same', input_shape=(5,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=52, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=18, kernel_size=(2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath_new = '../data/modelCheckpoint/k54_d7_1_{epoch:02d}-{val_loss:.4f}.hdf5'
# #02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
# #따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5
# es= EarlyStopping(monitor='val_loss', patience=64, mode='auto')
# cp =ModelCheckpoint(filepath=modelpath_new, monitor='val_loss',
#                     save_best_only=True, mode='auto')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)

model.compile(loss='mean_squared_error', 
              optimizer='rmsprop', metrics=['mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train,y_train, epochs=250, batch_size=16, verbose=1,
                 validation_split=0.3) #,callbacks=[es, cp]

# model.save('../data/h5/samsung8_model.h5')

#4. training
loss, mae = model.evaluate(x_test, y_test, batch_size=16) # batch_size뜻 데이터를 16번 반복해서 돌린다.
print('loss, mae: ', loss, mae)

x_predict = np.array([[88700,90000,88700,26127127,2332653]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1) #x와 차원
y_predict = model.predict(x_predict)
print(y_predict)

