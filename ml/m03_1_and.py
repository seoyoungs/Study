### AI의 겨울
from  sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score #accuracy_score만 따로 메트릭스로 빼줌

#1. 데이터
x_data = [[0,0], [1,0],[0,1], [1,1]]
y_data = [0,0,0,1]

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가예측
result = model.score(x_data, y_data) #accuracy model.score : 1.0 ---> 100%일치
print(' model.score :', result)

y_pred = model.predict(x_data)
print(x_data, '의 예측결과:', y_pred)

acc = accuracy_score(y_data, y_pred) #accuracy_score : 1.0  =model.score 동일하다
print('accuracy_score :', acc)
# model.score, accuracy_score 두개 다 사용가능하다. 

