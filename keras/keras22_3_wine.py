#다중분류

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
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1,1)).toarray()
print(y)

###데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

'''
######tensorflow로 전처리 하기 방법
## y사 1,2,3이니 원핫 인코딩 해야한다.
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(178, 3)
#print(y)
'''
#print(x_train.shape) #(142, 13)
#print(x_test.shape) #(36, 13)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(13,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)

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
y1_pred = model.predict(x_test[-5:-1])
print(y1_pred)
#print(np.argmax(y1_pred, axis=1))


'''
LSTM하기 전
[0.056005265563726425, 0.9722222089767456, 0.024329908192157745]
[[9.9724627e-01 3.5298237e-04 2.4007882e-03]
 [8.2857476e-04 3.2784534e-04 9.9884355e-01]
 [9.9616253e-01 1.1043251e-03 2.7331631e-03]
 [2.3583854e-02 9.7291708e-01 3.4989906e-03]]
'''