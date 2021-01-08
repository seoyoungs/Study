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

######sklearn로 전처리 하기 방법2
## y사 1,2,3이니 원핫 인코딩 해야한다.
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#print(y.shape) #(178, 3)
#print(y)

#print(x_train.shape) #(142, 13)
#print(x_test.shape) #(36, 13)
x_train=x_train.reshape(142,13,1)
x_test=x_test.reshape(36,13,1)

#2. 모델
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(13,1)))
model.add(Dense(5, activation='relu')) #y값 3개이다(0,1,2)
model.add(Dense(6, activation='relu')) #y값 3개이다(0,1,2)
model.add(Dense(3, activation='softmax')) #y값 3개이다(0,1,2)

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc', 'mae'])
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
LSTM 한 후
 [0.3149387836456299, 0.8333333134651184, 0.14735256135463715]
[[9.8698944e-01 1.3002690e-02 7.8834273e-06]
 [1.4984495e-03 6.7161453e-01 3.2688701e-01]
 [9.6040803e-01 3.9365381e-02 2.2655104e-04]
 [1.2893325e-05 3.3349603e-01 6.6649103e-01]]
[0.0737428367137909, 0.9722222089767456, 0.03970159962773323]
[[9.6777380e-01 3.2193895e-02 3.2290594e-05]
 [2.5226389e-14 2.7594838e-04 9.9972409e-01]
 [9.7183585e-01 2.8143577e-02 2.0625928e-05]
 [2.5473499e-01 7.4396855e-01 1.2964138e-03]]
 
 LSTM ---> 결과가 더 안좋다.
'''
