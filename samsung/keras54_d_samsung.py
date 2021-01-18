import pandas as pd
import matplotlib.pyplot as plt

sam = pd.read_csv('C:\data\csv\samsung.csv', encoding='cp949', index_col=0, header=0)
#print(sam.head())

sam1= sam.loc[::-1] # 데이터 일자 역순으로 뒤집기
#print(sam1.head())

sam2=sam1[1738:]  #날짜 자르기
#print(sam2.head())
#print(sam2.tail())
sam3=sam2.iloc[:,0:6] # 열자르기
sam3 = sam3.drop(columns='등락률') # 열 등락률 삭제
#print(sam3.head())


x= sam3[['시가', '고가', '저가', '종가', '거래량']].values
y= sam3[['종가']].values

#x[i][j]를 string 타입으로 바꿔주는 걸 먼저 해줌(, ---이거 콤마 없애기)
for i in range(len(x)):
    for j in range(5):
        x[i][j] = str(x[i][j])
        x[i][j] = x[i][j].replace(',', '')
        y[i][0] = y[i][0].replace(',', '')

x_shift = x[:][0:-1]
y_shift = y[:][1:] 

#print(x.shape, y.shape, x_shift.shape, y_shift.shape)
#(662, 5) (662, 1) (661, 5) (661, 1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_shift, y_shift, 
                                              test_size = 0.3, shuffle=False)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(462, 5) (199, 5) (462, 1) (199, 1)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

lookback = 20 #x의 5일치 데이터로 자른다.

x_train_sequence = []
y_train_sequence = []

for i in range(len(y_train)-lookback+1):
    x_train_sequence.append(x_train[i:i+lookback])
    y_train_sequence.append(y_train[i+lookback-1])

#print(x_train_sequence[0], y_train_sequence[0])

#데이터 타입 바꾸기
type(x_train_sequence)
import numpy as np
x_input = np.array(x_train_sequence)
y_input = np.array(y_train_sequence)
# print(type(x_input), type(y_input))
# print(x_input.shape, y_input.shape) #(458, 5, 5) (458, 1)

# print(x_train.shape) #(462, 5)
# print(y_train.shape) #(462, 1)
# print(x_test.shape) #(199, 5)
# print(y_test.shape) #(199, 1)


x_train = x_train.reshape(-1, 1, 5).astype('float32')
x_test  = x_test.reshape(-1, 1, 5).astype('float32')
y_train = y_train.reshape(-1, 1, 1).astype('float32')
y_test = y_test.reshape(-1, 1, 1).astype('float32')
print(x_train.shape, x_test.shape)
'''
#2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, SimpleRNN

model=Sequential()
model.add(LSTM(12, activation='relu', input_shape=(20,5))) #이거 하나만 하면 훈련값 제대로 안나온다.
model.add(Dense(4, activation='relu'))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_data=(x_test, y_test), verbose=1)

#4. 평가예측
loss, mae=model.evaluate(x_test, y_test,batch_size=10)
print('loss:', loss)
print('mae :', mae)
x_test = np.array([x_test[0]])
pred = model.predict(x_test)
print('pred : ', pred)
print(pred.flatten()[0])
'''





