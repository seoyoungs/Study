import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as k

#mse 커스팅 mse하려면 미리 이렇게 인자 만들어주기
def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
  # reduce_mean(tf.squredy(y_true - y_pred)) = MSE

def quantile_loss(y_true, y_pred):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #텐서플로우의 함수형식
    q= tf.constant(np.array([qs]), dtype= tf.float32)
    e= y_true - y_pred
    v= tf.maximum(q*e, (q-1)*e)
    return k.mean(v) # mean이므로 평균값이다.
    #이거를(하나하나 9개 나오게 커스텀해보기) 그거를 submission에 제출
    # 여기서 0.5는 중위값으로 mae, mse랑 비슷

def quantile_loss(q, y_true, y_pred):
      err = (y_true - y_pred)
      return k.mean(k.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #quantiles이 함수밖

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
model.compile(loss= quantile_loss, optimizer='adam')
model.compile(loss = lambda y_true, y_pred: quantile_loss(quantiles[0], y_true, y_pred), optimizer='adam')
#y_true, y_pred 인풋값, quantiles[0] 이면 0.1 한번 돌아감 'for문' 으로 돌리기
model.fit(x, y, batch_size=1, epochs=30)

# 4. 평가
loss = model.evaluate(x,y)
print(loss)

'''
custom_mse
0.08615630120038986

quantile_loss
0.007723234593868256

quantile_loss ==> 0.1
0.06900101155042648
'''

