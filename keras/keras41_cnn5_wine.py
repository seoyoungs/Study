from sklearn.datasets import load_wine

dataset = load_wine()
#print(dataset.DESCR)  #열 13개 y 3개
#print(dataset.feature_names)

x=dataset.data
y=dataset.target

'''
print(x)
print(y) #0,1,2 다중분류
print(x.shape) #(178,13)
print(y.shape) #(178, )
'''

###실습, DNN완성
'''
######sklearn로 전처리 하기 방법1
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
e = LabelEncoder()
e.fit(y)
y1 = e.transform(y)
y1 = np_utils.to_categorical(y1)
print(y1.shape)
print(y1)
'''
###데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
#print(x_train.shape) #(142, 13)
#print(x_test.shape) #(36, 13)
x_train=x_train.reshape(142,13,1,1)
x_test=x_test.reshape(36,13,1,1)

######sklearn로 전처리 하기 방법2
## y사 1,2,3이니 원핫 인코딩 해야한다.
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(178, 3)
#print(y)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1), 
                padding='same', strides=(1,1), input_shape=(13,1,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Dense(4, activation='relu'))
model.add(Flatten())
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(3))
model.add(Dense(3, activation='softmax')) 

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
model.fit(x_train,y_train, epochs=150, 
           validation_split=0.2, batch_size=10,verbose=1)

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print(loss)
'''
y1_pred = model.predict(x_test[-5:-1])
print(y1_pred)
#print(np.argmax(y1_pred, axis=1))
'''
'''
LSTM하기 전
[0.056005265563726425, 0.9722222089767456, 0.024329908192157745]

LSTM 한 후
 [0.3149387836456299, 0.8333333134651184, 0.14735256135463715]
[0.0737428367137909, 0.9722222089767456, 0.03970159962773323]

CNN
loss: 0.1033 - acc: 0.9722
[0.10332775115966797, 0.9722222089767456]
 '''