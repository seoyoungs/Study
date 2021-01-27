# 회귀모델

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
# from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression #이것만 분류에 해당
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터
dataset = load_boston()

x= dataset.data
y = dataset.target


#2. 모델
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

model.fit(x,y)



result = model.score(x, y) #evaluate 대신 score사용
# print(result)  #loss, accurac 값 추출
y_pred=model.predict(x)
# print(y_pred) # y_pred로 코딩한 값

result = model.score(x, y) #accuracy model.score : 1.0 ---> 100%일치
print('model.score :', result)


r2 = r2_score(y, y_pred) #이게 회귀모델에서는 model.score 대신 사용
print('r2_score :', r2)
# 여기서는 r2_score로 한다.

'''
봐야할 것 model들의 차이점, 성능차이

model = LinearRegression()
model.score : 0.7406426641094094
r2_score : 0.7406426641094094
이렇게 동일하게 나온다.

model = KNeighborsRegressor()
model.score : 0.716098217736928
r2_score : 0.716098217736928

model = DecisionTreeRegressor()
model.score : 1.0
r2_score : 1.0

model = RandomForestRegressor()
model.score : 0.9828473102388375
r2_score : 0.9828473102388375

--------------------------------
tensorflow
R2 :  0.8125122028343794
'''

