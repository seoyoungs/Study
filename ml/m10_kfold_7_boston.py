# 회귀모델

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
# from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression, LogisticRegression #이것만 분류에 해당
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
dataset = load_boston()

x= dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True) 

#2. 모델
model_list = {'최근접 이웃':KNeighborsRegressor(),
               '로지스틱 회귀':LinearRegression(),
              '결정 트리':DecisionTreeRegressor(),
              '랜덤 포레스트':RandomForestRegressor()}


for model_name, model in model_list.items():
    scores = cross_val_score(model, x, y, cv = kflod) #fit이랑 model이랑 다 포함되있는것(단, validation은 안 나눠짐)
    print('scores :', scores, model_name)

# model.fit(x,y)
# result = model.score(x, y) #evaluate 대신 score사용
# print(result)  #loss, accurac 값 추출
# y_pred=model.predict(x)
# print(y_pred) # y_pred로 코딩한 값

# result = model.score(x, y) #accuracy model.score : 1.0 ---> 100%일치
# print('model.score :', result)


# r2 = r2_score(y, y_pred) #이게 회귀모델에서는 model.score 대신 사용
# print('r2_score :', r2)
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
for문으로 Kflod
scores : [0.62815988 0.53994074 0.43395404 0.50978793 0.5709981 ] 최근접 이웃
scores : [0.70966967 0.73966244 0.7432015  0.69832587 0.71079372] 로지스틱 회귀
scores : [0.50297731 0.72556726 0.76674754 0.66672732 0.7687922 ] 결정 트리
scores : [0.84331254 0.89566502 0.88409628 0.76370488 0.84462536] 랜덤 포레스트
'''