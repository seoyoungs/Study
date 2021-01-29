##  머신러닝으로 해보기
##keras22_iris사용
### m04_iris 복붙
# KFold, cross_val_score (교차검증을 해 KFold구한다.)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import warnings
warnings.filterwarnings(action='ignore')

dataset = load_iris() #x,y=load_iris(return_X_y=True)와 같다
x= dataset.data
y= dataset.target
#print(dataset.DESCR)
#print(dataset.feature_names)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kfold = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

# 2,3. 모델링과 훈련

model_list = {'최근접 이웃':KNeighborsClassifier(),
               '로지스틱 회귀':LogisticRegression(),
              '선형 SVM':LinearSVC(),
              '비선형 SVM':SVC(),
              '결정 트리':DecisionTreeClassifier(),
              '랜덤 포레스트':RandomForestClassifier()}

for (model_name, model) in model_list.items():
    scores = cross_val_score(model, x, y, cv = kfold) #fit이랑 model이랑 다 포함되있는것(단, validation은 안 나눠짐)
    print('scores :', scores, model_name)

'''
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
kflod 일 때
scores : [0.96666667 1.         0.96666667 0.86666667 1.        ]
for 문으로 묶었을 때
warning무시하는 것 넣기
import warnings
warnings.filterwarnings(action='ignore')
scores : [0.96666667 0.96666667 0.9        0.96666667 0.96666667] 최근접 이웃
scores : [1.         0.96666667 0.96666667 0.93333333 0.93333333] 로지스틱 회귀
scores : [0.9        0.93333333 0.96666667 0.96666667 0.96666667] 선형 SVM
scores : [0.96666667 0.96666667 1.         0.96666667 0.93333333] 비선형 SVM
scores : [0.96666667 0.93333333 0.93333333 0.96666667 0.96666667] 결정 트리
scores : [1.         0.9        0.96666667 0.96666667 0.93333333] 랜덤 포레스트
'''
