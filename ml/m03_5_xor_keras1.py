### 딥러닝 형식으로 풀기

### AI의 겨울
from  sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score #accuracy_score만 따로 메트릭스로 빼줌
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0], [1,0],[0,1], [1,1]]
y_data = [0,1,1,0] # or 니까 1로 바뀐다. 같거나 같지않아도 되므로


#2. 모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) #y가 0,1 이진법이므로 sigmoid
#인풋과 아웃풋만 쓰기(히든 레이어)


#3. 훈련
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가예측
y_pred = model.predict(x_data)
print(x_data, '의 예측결과:', y_pred)

result = model.score(x_data, y_data) #accuracy model.score : 1.0 ---> 100%일치
print('model.score :', result[1])

acc = accuracy_score(y_data, y_pred) #accuracy_score : 1.0  =model.score 동일하다
print('accuracy_score :', acc)
# model.score, accuracy_score 두개 다 사용가능하다. 

'''
xor이면 
 model.score : 0.75
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과: [0 1 1 1]
accuracy_score : 0.75 가 최대이다
[1,1]이면 
어떻게 해야 acc 가 1이 나올까/

model = SVC() 사용
 model.score : 1.0
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과: [0 1 1 0]
accuracy_score : 1.0
'''

