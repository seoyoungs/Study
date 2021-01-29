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


dataset = load_iris() #x,y=load_iris(return_X_y=True)와 같다
x= dataset.data
y= dataset.target
#print(dataset.DESCR)
#print(dataset.feature_names)

kfold = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

# 2,3. 모델링과 훈련
model = LinearSVC() #머신러닝은 이거 하나면 된다.
scores = cross_val_score(model, x, y, cv = kfold) #fit이랑 model이랑 다 포함되있는것(단, validation은 안 나눠짐)
print('scores :', scores)
# scores : [0.96666667 1.  0.96666667 0.86666667 1.   ] -> 5개 나온다.(fit까지 깔끔하게 나온다)

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
'''

