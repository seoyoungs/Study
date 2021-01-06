#Dense모델

#과제 실습
#LSTM, early, 전처리 다하기
#데이터 1~100
#    x            y
# 1,2,3,4,5       6
#.....
# 95,96,97,98,99  100

#predict를 만들것
#96, 97,98,99,100   101
#....
#100, 101, 102, 103, 104  ->105
##총 5개의 predict
#예상되는 predict는 (101, 102, 103, 104, 105)  ---(5,5)행렬


import numpy as np

a = np.array(range(1,101))
size=6

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

x = dataset[:,:5]
y = dataset[:,-1]  ###행렬은 : 로 구분한다.
#y=dataset[0:-1,-1]


print(x.shape) # (95, 5) -> LSTM(95,5,1)
print(y.shape)  # (95,)'
#x=x.reshape(95,5,1)

#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(33, activation = 'relu', input_dim=5))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
# LSTM을 DNN으로 구현 가능

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100, batch_size=5, verbose=1)

#4. 평가, 예측
loss= model.evaluate(x,y, batch_size=5)
print('loss :', loss)


b=np.array(range(96,105))
x_pred=split_x(b, 5)
x_pred=scaler.transform(x_pred)
#x_pred=x_pred.reshape(5,5,1)

y_predict=model.predict(x_pred)
print(y_predict)




