import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as k

#mse 커스팅 mse하려면 미리 이렇게 인자 만들어주기
def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
  # reduce_mean(tf.squredy(y_true - y_pred)) = MSE

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') 
#.astype('float32')안하려면 1., 2. 점부치기
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')
print(x.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일
model.compile(loss= custom_mse, optimizer='adam')
model.fit(x, y, batch_size=1, epochs=30)

# 4. 평가
loss = model.evaluate(x,y)
print(loss)

'''
custom_mse
0.08615630120038986

quantile_loss
0.007723234593868256
'''






