##  머신러닝으로 해보기
##keras22_iris사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

model = DecisionTreeClassifier() #머신러닝은 이거 하나면 된다.

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

'''
model =LinearSVC()일 때
0.9666666666666667

model =SVC()일 때
0.9733333333333334

model =KNeighborsClassifier()일 때
0.9666666666666667

model =DecisionTreeClassifier()일 때
1.0

model = DecisionTreeClassifier()일 때
1.0

----------------
tensorflow
keras33과 비교
0.9666666388511658
'''

