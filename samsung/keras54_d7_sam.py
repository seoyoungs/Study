import numpy as np
import pandas as pd

x = np.load('../data/npy/sam2.npy',allow_pickle=True)[0]
y = np.load('../data/npy/sam2.npy',allow_pickle=True)[1]
x_pred = np.load('../data/npy/sam2.npy',allow_pickle=True)[2]

x_pred = x_pred.reshape(-1,5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM, SimpleRNN
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=(2), padding='same', input_shape=(5,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=150, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=65, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=18, kernel_size=(2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='tanh'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.compile(loss='mae', 
              optimizer='rmsprop', metrics=['mae']) #optimizer='rmsprop'
hist = model.fit(x_train,y_train, epochs=200, batch_size=32, verbose=1,
                 validation_split=0.3,callbacks=[es])

#4. training
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
print('loss, mae: ', loss, mae)

x_predict = np.array([[89800,91200,89100,34161101,3073003]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1) #x와 차원
y_predict = model.predict(x_predict)
print(y_predict)

