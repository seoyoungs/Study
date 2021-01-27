#이중분류

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_breast_cancer()

x= dataset.data
y= dataset.target

# model =LinearSVC()
# model =SVC()
# model =KNeighborsClassifier()
# model = DecisionTreeClassifier()
model =DecisionTreeClassifier()

model.fit(x, y)

#4. 평가 ,예측
# result =model.evaluate(x_test,y_test, batch_size=10)
result = model.score(x, y) #evaluate 대신 score사용
print(result)  #loss, accurac 값 추출
# y_pred=model.predict(x[-5:-1])
# print(y_pred) # y_pred로 코딩한 값
# print(y[-5:-1]) 

'''
model =LinearSVC()일 때
0.9314586994727593

model =SVC()일 때
0.9226713532513181

model =KNeighborsClassifier()일 때
0.9472759226713533

model =DecisionTreeClassifier()일 때
1.0

model = DecisionTreeClassifier()일 때
1.0

---------------
tensorflow
keras33과 비교
0.9824561476707458
'''




