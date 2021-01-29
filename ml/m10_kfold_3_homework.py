###keras22_iris사용, m04_iris 복붙
# KFold, cross_val_score (교차검증을 해 KFold구한다.)
#과제 : train, test 나눈다음에 train만 발리데이션 하지말고 
#       kflod한 후에 train_test_split사용 (즉, 5등분 후 2등분)=>val생성

###==========================================================================================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris() #x,y=load_iris(return_x_y=True)와 같다
x= dataset.data
y= dataset.target
#print(dataset.DESCR)
#print(dataset.feature_names)

kflod = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

for train_index, test_index in kflod.split(x): # kflod 한 후에 train, test 나눈후 그 후에 train 값에서 val추출하기
    print('===================================')
    print("TRAIN:", train_index, "\nTEST:", test_index) # train, test 각 5번씩 나눠짐, n_splits=5 이므로
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    #train, test, validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                     train_size=0.8, random_state = 77, shuffle=False)


# 2,3. 모델링과 훈련
model = LinearSVC() #머신러닝은 이거 하나면 된다.
scores = cross_val_score(model, x_train, y_train, cv = kflod) #fit이랑 model이랑 다 포함되있는것(단, validation은 안 나눠짐)
print('scores :', scores) 

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
kfold먼저 한 후 train, test 나눈후 train에서 validation더 부여
#scores : [0.95       1.         0.94736842 0.94736842 0.89473684]
'''
