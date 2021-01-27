##  머신러닝으로 해보기
##keras22_iris사용


import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

dataset = load_iris() #x,y=load_iris(return_X_y=True)와 같다
x= dataset.data
y= dataset.target
#print(dataset.DESCR)
#print(dataset.feature_names)

#print(x.shape) #(150,4)
#print(y.shape) #y가 3종류(150,)---> 바뀔거다
#print(x[:5])

# from sklearn.preprocessing import OneHotEncoder 
# #sklearn 데이터 다중분류 경우 이거 사용
# encoder = OneHotEncoder()
# y = encoder.fit_transform(y.reshape(-1,1)).toarray()

# # 전처리 알아서/ minmax, train_test_split
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test= train_test_split(x,y, 
#                      shuffle=True, train_size=0.8, random_state=66)

# from sklearn.preprocessing import MinMaxScaler
# scaler =MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(3))
# model.add(Dense(3))
# model.add(Dense(3, activation='softmax'))

model =LinearSVC() #머신러닝은 이거 하나면 된다.

#3. 컴파일, 훈련
#                    #mean_squared_error
# model.compile(loss='categorical_crossentropy', 
#               optimizer='adam', metrics=['acc', 'mae'])
# ####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
# model.fit(x_train,y_train, epochs=150, 
#            validation_split=0.2, batch_size=10,verbose=0)
model.fit(x, y)

#4. 평가 ,예측
# result =model.evaluate(x_test,y_test, batch_size=10)
result = model.score(x, y) #evaluate 대신 score사용
print(result)  #loss, accurac 값 추출
y_pred=model.predict(x[-5:-1])
print(y_pred) # y_pred로 코딩한 값
print(y[-5:-1]) 

##argmax로 결과치 수정

# y1_pred = model.predict(x_test[-5:-1])
# print(y1_pred)
# print(np.argmax(y1_pred, axis=1))# 가장 큰 클래스를 출력해주는 함수
# #axis=0 0이 위치하는 데이터 인덱스가 0부터 시작하므로 [0,1,2]
# #행마다 위치하는 값이 2니까 행이 

'''
결과
[0.2929254472255707, 0.9666666388511658, 0.15748263895511627]
[[1.4803344e-03 8.8146053e-02 9.1037357e-01]
 [9.9200785e-01 7.4190465e-03 5.7309697e-04]
 [7.6894391e-01 2.2478288e-01 6.2731630e-03]
 [1.1676942e-02 5.5393237e-01 4.3439060e-01]]
[2 0 0 1]

머신 러닝 일 때 -> 속도, 정확도 높음
0.9666666666666667 -> 정확도 바로 반환(accuracy)
[2 2 2 2]
[2 2 2 2]
'''