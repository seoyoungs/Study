import numpy as np
import tensorflow as tf
#4단계의 모델 제작예정 1,2,3,4보기
#1. 데이터
x= np.array([1,2,3])
y= np.array([1,2,3]) #x,y 개수 같아야 한다. 값이 너무 다르면 이상치라고 함

#2.모델구성
from tensorflow.keras.models import Sequential 
#tensorflow에 keras에 models에 Sequential(순차적)라는 것을 불러오겠다는 뜻
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
# input_dim=1, 한개의 dim(차원)을 두겠다는 뜻 -- 첫번째 layer 5개
model.add(Dense(3, activation='linear'))
#두번째 input표시 안하는 이유- 앞에 Sequential을 명시하고 model.add로 묶어서
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#loss 최소값으로 하고 optimizer을 adam으로 할 것
model.fit(x,y,epochs=100, batch_size=1) 
#훈련시킬 x,y를 100번 훈련, batch_size=1 :100번세트를 한 번 훈련시키자

#4. 평가예측
loss = model.evaluate(x,y, batch_size=1) 
# model.evaluate평가하겠다. x,y로 또한 loss값으로 반환한다.
print('loss : ', loss) #loss출력
# result=model.predict([4]) #4에대한 예측 결과값
x_pred = np.array([4]) #이렇게 해도 위에 result처럼 4에대한 값 나온다.
result = model.predict(x_pred)
#result=model.predict([x]) --> 만약이렇게 하면  x에 대한 예측된 y값이 나온다.
print('result : ', result)