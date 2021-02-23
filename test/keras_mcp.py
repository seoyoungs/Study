'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.reshape[0], -1)/255.
x_test = x_test.reshape(x_test.reshape[0], -1)/255.
#print(x_train.shape)

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(Dense(12, activation='relu', input_shape=(28*28,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath= './modelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath='modelpath1', monitor='val_loss', 
                    save_best_only=True, auto='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
hist=model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=1,
               validation_split=0.2, callbacks=[es,cp])

#4. 평가, 훈련
loss=model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)

y_pred= model.predict(x_test[:10])
print('y_pred : ', y_pred.argmax(axis=1))
print('y_test : ', y_test.argmax(axis=1))

####시각화
import matplotlib.pyplot as plt
plt.figure
'''

import numpy as np
from sklearn.datasets import load_boston 

# Data
boston = load_boston()
x = boston.data
y = boston.target 

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)
print(np.min(x), np.max(x)) # 0.0 711.0   

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128,activation = 'relu',input_dim = 13),
    # model.add(Dense(10, activation='relu',input_shape=(13,))
    Dense(128),
    Dense(64),
    Dense(64),
    Dense(32),
    Dense(32),
    Dense(16),
    Dense(16),
    Dense(1),
])

# Compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Earlystooping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = './modelCheckpoint/k45_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss',patience = 20,mode = 'auto')
check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

# fit
hist = model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_split=0.2, callbacks=[early_stopping,check_point])

#4. Evaluate
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("mae : ", mae)

# prediction
y_predict = model.predict(x_test)
# print("y_pred : \n", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test, y_train))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)

# visualization
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
plt.subplot(211)    # 2 row 1 column
plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
plt.grid()

plt.title('Cost')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()