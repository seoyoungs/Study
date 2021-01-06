###  https://ebbnflow.tistory.com/135 이분꺼 참조하기
# 모델을 구성하시오
#######DENSE 모델만


import numpy as np

a = np.array(range(1,11))
size=5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] #위치 잘 맞춰서 넣어주기
        aaa.append(subset)
        #aaa.append([item for item  in subset])
    print(type(aaa))
    return np.array(aaa)
 
dataset = split_x(a, size)
#print('------------------')
#print(dataset)

x = dataset[:,0:4]
y = dataset[:,-1]  ###행렬은 : 로 구분한다.
#y=dataset[0:-1,-1]
'''
#print(x.shape) # (6, 4) -> LSTM(6,4,1)
#print(y.shape)  # (6,)

print(x) 
print(y)  # (6,)
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
[ 5  6  7  8  9 10]  --> y나중에 predict할 때 비슷하게 나와야함
'''
#x=x.reshape(6,4,1)
#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(33, activation = 'relu', input_dim=4))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
# LSTM을 DNN으로 구현 가능

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=10, batch_size=1, verbose=1)

#4. 평가, 예측
loss= model.evaluate(x,y)
print('loss :', loss)
y_predict=model.predict(x)
print(y_predict)

'''
LSTM으로 했을 때
loss: 0.00984026025980711
[[4.8213367]
 [5.9702363]
 [7.0560813]
 [8.068698 ]
 [8.999465 ]
 [9.864463 ]]

 Dense로 했을 때
loss : 9.686595916748047
[[1.526029 ]
 [2.6889782]
 [3.8412576]
 [4.9803534]
 [6.108434 ]
 [7.2375436]]
'''