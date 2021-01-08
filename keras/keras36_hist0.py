####회귀
###따라서 회귀라 acc(정확도) 안맞음, loss와 val_loss만 보기

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
a= np.array(range(1,101))
size=5

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] #위치 잘 맞춰서 넣어주기
        aaa.append(subset)
        #aaa.append([item for item  in subset])
    print(type(aaa))
    return np.array(aaa)

dataset= split_x(a, size)
print(dataset.shape)


x = dataset[:,0:4]
y = dataset[:,-1]
print(x.shape, y.shape) #(96,4), (96, )

x= x.reshape(x.shape[0], x.shape[1], 1)
print(x) #(96,4,1)

#2.모델
model = load_model('./model/save_keras35.h5')
model.add(Dense(5, name='kingkeras1')) #이름 같아도 상관없음
model.add(Dense(1, name='kingkeras2'))

#얼리 스탑핑 적용
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='loss', patience=10, mode='auto')

#컴파일
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x,y, epochs=1000, batch_size=32, verbose=1, 
         validation_split=0.2, callbacks=[es])

print(hist)
print(hist.history.keys())
#print(hist.history['loss']) ###로스값이 차례대로 줄어드는 것을 볼 수 있다.

'''
print(hist.history.keys())하면
metrics=['acc'] 추가전
#dict_keys(['loss', 'val_loss'])
metrics=['acc'] 추가전 추가후
dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
'''
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
###그래프 찍힌것 y을 대상으로 찍힌 것이다.
###그림보면 val_loss는 항상loss 보다 낮다.
###아니면 val_loss와 loss가 너무 벌어져도 과적합이다
##신뢰도가 낮음
