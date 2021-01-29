# sklearn회귀모델

import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score #(회귀일때)
# from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression #이것만 분류에 해당
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

'''
model = LinearRegression()
model.score : 0.7406426641094094
r2_score : 0.7406426641094094
model = KNeighborsRegressor()
model.score : 0.716098217736928
r2_score : 0.716098217736928
model = DecisionTreeRegressor()
model.score : 1.0
r2_score : 1.0
model = RandomForestRegressor()
model.score : 0.983719108568683
r2_score : 0.983719108568683
---------------------------------------
tensorflow
R2 : 0.5754409282584073
for문 kflod
scores : [0.58433932 0.50987625 0.33417758 0.56773536 0.5693401 ] 최근접 이웃
scores : [0.80137891 0.73968905 0.55029773 0.71779532 0.76045115] 로지스틱 회귀
scores : [0.56186688 0.8951556  0.75760359 0.73079183 0.7036751 ] 결정 트리
scores : [0.90373512 0.72152091 0.85115817 0.89015564 0.91218142] 랜덤 포레스트
'''
