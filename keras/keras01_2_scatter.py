from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,10])
print('\n',x,'\n',y) # 줄맞추기

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_shape=(1,))) 
# 머신러닝은 히든layer없다. 그래서 epochs 많이 잡음, 
# epochs많이 잡아도 딥러닝보다 연산량적다
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))
# 이렇게 하는 것은 딥러닝 형식(가중치 형식)

# 3. 컴파일 훈련
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='mse', optimizer= optimizer)
model.fit(x,y, epochs=10)

y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x, y_pred, color= 'red')
plt.show()
