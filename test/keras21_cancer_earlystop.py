import numpy as np
from sklearn.datasets import load_breast_cancer

#1.데이터
datasets = load_breast_cancer()

#print(datasets.DESCR)
#print(datasets.feature_names)

x= datasets.data
y= datasets.target
#print(x.shape) #(569,30) # 실질적 칼럼개수(32개, id와 y칼럼빼고)
#print(y.shape) #(569,)
#y값은 diagnosis 진단 여부(B(양성), M(악성))
#print(x[:5])
#print(y)

# 전처리 알아서/ minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, 
                    shuffle=True, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x,y,
                  shuffle=True, train_size=0.8, random_state=66) 


from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
#model.add(Dense(1)) #-->hidden이 없는 layer가 있다.
model.add(Dense(10, activation='relu')) 
# activation 다음 층으로 연결할때 전달하는 방법
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #마지막에만 sigmoid를 준다
#input, output에 sigmoid를 줘도 된다. 각자의 판단을 함수 적용

#3. 컴파일, 훈련
                   #mean_squared_error
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['acc', 'mae'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=250, batch_size=10, 
          validation_data=(x_val, y_val), verbose=0, callbacks=[early_stopping])

#4. 평가 ,예측
loss=model.evaluate(x_test,y_test, batch_size=10)
print(loss)  #loss, accurac 값 추출
y_pred=model.predict(x_test[-5:-1])
print(np.argmax(y_pred, axis=1))
print(y_pred) # y_pred로 코딩한 값
print(y_test[-5:-1]) #원래 기존 y값
#y_predict=model.predict(x_test)
#print(y_predict) #왜 소수점이야?

'''
결과
[0.08244305104017258, 0.9824561476707458, 0.034643013030290604]
[0 0 0 0]
[[9.9981827e-01]
 [1.3578001e-04]
 [9.9937433e-01]
 [9.9968088e-01]]
[1 0 1 1]
'''
