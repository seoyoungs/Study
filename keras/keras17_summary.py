import numpy as np
import tensorflow as tf

#1.데이터
x= np.array([1,2,3])
y= np.array([1,2,3])

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(6, input_dim=1, activation='linear'))
model.add(Dense(5, activation='linear'))
model.add(Dense(8, name='aaaa'))
model.add(Dense(1))
model.summary()

'''
#실습
과제1. ensemble1,2,3,4 에 대해 서머리를 계산하고 
이해한 것을 과제로 제출하기 (일일히 서머리하고 제출)
과제2 
layer를 만들 때 'name'이란 것에 대해 확인하고 
왜 레이어에 이름을 짓는지 반드시 필요한 경우가 무엇인지
앙상블, 서머리 배운것 알기
'''



