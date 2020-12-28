import numpy as np
import tensorflow as tf

x= np.array([1,2,3])
y= np.array([1,2,3])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=1)

#평가예측
loss = model.evaluate(x,y,batch_size=1)
print('loss : ', loss)
result=model.predict([4])
print('result : ', result)