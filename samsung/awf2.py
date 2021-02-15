import numpy as np


x = np_load = np.load('../data/npy/x_awf_data.npy')
y = np_load = np.load('../data/npy/y_awf_data.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#print(x_train.shape, x_test.shape)

#모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout,LSTM

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(2), padding='same', input_shape=(7,1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=52, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=(2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=18, kernel_size=(2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# model.summary()


#컴파일, 훈련
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=200, batch_size=20, validation_batch_size=0.3, verbose=1, callbacks=[early_stopping])

loss, mae = model.evaluate(x_test, y_test, batch_size=20)
print('loss, mae: ', loss, mae)


x_predict = np.array([[24050,25850, 22350,731094,17307943550,239031278400, 10128444]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(x_predict.shape[0],x_predict.shape[1],1) 

y_predict = model.predict(x_predict)

print(y_predict)
'''
종가
25318.92, 23501.998, 25475.719, 24420.756, 24366.941, 24391.992

시가
24062.162, 25403.855, 23193.8

고가
25289.963, 24952.021, 24195.713, 27129.838, 
27708.596, 26164.506, 26136.898
'''

