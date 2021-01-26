import numpy as np

#1. data
x= np.array([1,2,3,4,5,6,7,8,9,10])
y= np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. model 
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. compile 
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer= Nadam(lr= 0.1) #learning late 간격조절
### 옵티마이저는 그라디언트 (경사하강법)에 기반 learning late 하강시키면서함
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4. evaluate, predict
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print('loss : ', loss, '결과물 :', y_pred)

'''
<Adam>
optimizer= Adam(lr= 0.1)
loss :  2.5563244889781345e-06 결과물 : [[10.998582]]
optimizer= Adam(lr= 0.01)
loss :  3.6014227156044853e-09 결과물 : [[11.000081]]
optimizer= Adam(lr= 0.001)
loss :  2.870592760891022e-13 결과물 : [[10.999997]]
optimizer= Adam(lr= 0.0001)
loss :  1.9653773506433936e-06 결과물 : [[10.997501]]

<Adadelta>
optimizer= Adadelta(lr= 0.1)
loss :  0.0004461708012968302 결과물 : [[11.0443535]]
optimizer= Adadelta(lr= 0.01)
loss :  0.0015306600835174322 결과물 : [[11.065936]]
optimizer= Adadelta(lr= 0.001)
loss :  12.840377807617188 결과물 : [[4.583078]]
optimizer= Adadelta(lr= 0.0001)
loss :  25.13136863708496 결과물 : [[2.1051269]]

<Adamax>
optimizer= Adamax(lr= 0.1)
loss :  0.24897980690002441 결과물 : [[10.335611]]
optimizer= Adamax(lr= 0.01)
loss :  8.103740118184377e-13 결과물 : [[11.000001]]
optimizer= Adamax(lr= 0.001)
loss :  8.252986560819409e-08 결과물 : [[10.999462]]
optimizer= Adamax(lr= 0.0001)
loss :  0.0019505582749843597 결과물 : [[10.940265]]

<Adagrad>
optimizer= Adagrad(lr= 0.1)
loss :  2346.499267578125 결과물 : [[-45.14346]]
optimizer= Adagrad(lr= 0.01)
loss :  1.1210397588001797e-06 결과물 : [[11.002178]]
optimizer= Adagrad(lr= 0.001)
loss :  2.1373070921981707e-05 결과물 : [[10.991831]]
optimizer= Adagrad(lr= 0.0001)
loss :  0.004050609655678272 결과물 : [[10.920319]]


<RMSprop>
optimizer= RMSprop(lr= 0.1)
loss :  117685616.0 결과물 : [[-11540.077]]
optimizer= RMSprop(lr= 0.01)
loss :  24.044017791748047 결과물 : [[1.568438]]
optimizer= RMSprop(lr= 0.001)
loss :  0.0003462765016593039 결과물 : [[10.971601]]
optimizer= RMSprop(lr= 0.0001)
loss :  0.02033044397830963 결과물 : [[10.745045]]

<SGD>
optimizer= SGD(lr= 0.1)
loss :  nan 결과물 : [[nan]]
optimizer= SGD(lr= 0.01)
loss :  nan 결과물 : [[nan]]
optimizer= SGD(lr= 0.001)
loss :  3.1578013022226514e-06 결과물 : [[11.002354]]
optimizer= SGD(lr= 0.0001)
loss :  0.002370962407439947 결과물 : [[10.950725]]


optimizer= Nadam(lr= 0.1)
loss :  64339440566272.0 결과물 : [[13092518.]]
optimizer= Nadam(lr= 0.01)
loss :  2.0416088375441177e-08 결과물 : [[10.999721]]
optimizer= Nadam(lr= 0.001)
loss :  7.094004104146734e-05 결과물 : [[11.015468]]
optimizer= Nadam(lr= 0.0001)
loss :  4.567124506138498e-06 결과물 : [[11.000408]]
'''
