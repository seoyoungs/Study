##이진분류 --로지스틱 회귀넣기

#이중분류

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings(action='ignore')

dataset = load_breast_cancer()

x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

model_list = {'최근접 이웃':KNeighborsClassifier(),
               '로지스틱 회귀':LogisticRegression(),
              '선형 SVM':LinearSVC(),
              '비선형 SVM':SVC(),
              '결정 트리':DecisionTreeClassifier(),
              '랜덤 포레스트':RandomForestClassifier()}

for model_name, model in model_list.items():
    scores = cross_val_score(model, x, y, cv = kflod) #fit이랑 model이랑 다 포함(단, validation은 안 나눠짐)
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
for문으로 Kflod
scores : [0.94736842 0.95614035 0.97368421 0.88596491 0.9380531 ] 최근접 이웃
scores : [0.93859649 0.96491228 0.92105263 0.94736842 0.94690265] 로지스틱 회귀
scores : [0.9122807  0.93859649 0.92105263 0.92982456 0.9380531 ] 선형 SVM
scores : [0.9122807  0.94736842 0.89473684 0.93859649 0.87610619] 비선형 SVM
scores : [0.9122807  0.88596491 0.95614035 0.92982456 0.92920354] 결정 트리
scores : [0.93859649 0.98245614 0.96491228 0.92982456 0.98230088] 랜덤 포레스트
'''
