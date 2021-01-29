# 다중분류

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore')

dataset = load_wine()

x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True) 

model_list = {'최근접 이웃':KNeighborsClassifier(),
               '로지스틱 회귀':LogisticRegression(),
              '선형 SVM':LinearSVC(),
              '비선형 SVM':SVC(),
              '결정 트리':DecisionTreeClassifier(),
              '랜덤 포레스트':RandomForestClassifier()}


for model_name, model in model_list.items():
    scores = cross_val_score(model, x, y, cv = kflod) #fit이랑 model이랑 다 포함되있는것(단, validation은 안 나눠짐)
    print('scores :', scores, model_name)


# model.fit(x, y)

#4. 평가 ,예측
# result =model.evaluate(x_test,y_test, batch_size=10)
# result = model.score(x, y) #evaluate 대신 score사용
# print(result)  #loss, accurac 값 추출
# y_pred=model.predict(x[-5:-1])
# print(y_pred) # y_pred로 코딩한 값
# print(y[-5:-1]) 

'''
model =LinearSVC()일 때
0.634831460674157
model =SVC()일 때
0.7078651685393258
model =KNeighborsClassifier()일 때
0.7865168539325843
model =DecisionTreeClassifier()일 때
1.0
model = DecisionTreeClassifier()일 때
1.0
---------------------------
tensorflow
keras33과 비교
0.9722222089767456
for문으로 Kfold
scores : [0.69444444 0.66666667 0.69444444 0.68571429 0.62857143] 최근접 이웃
scores : [0.97222222 0.94444444 0.94444444 0.97142857 0.94285714] 로지스틱 회귀
scores : [0.63888889 0.97222222 0.91666667 0.88571429 0.94285714] 선형 SVM
scores : [0.58333333 0.63888889 0.69444444 0.68571429 0.74285714] 비선형 SVM
scores : [0.86111111 0.80555556 0.88888889 0.82857143 0.97142857] 결정 트리
scores : [0.97222222 0.91666667 0.97222222 0.97142857 1.        ] 랜덤 포레스트
'''
