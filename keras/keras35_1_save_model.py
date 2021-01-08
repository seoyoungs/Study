import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#모델 저장
model.save("./model/save_keras35.h5")  # ./ :이표시 현재 폴더
###현재 열려있는 위치는 study이다.
##근데 나중에 파이참으로 시험보는 것은 keras가 위치다 주의하기
###당분간 확장자 명은 .h5라 하자
'''
이거 다 저장된다.----> 아무거나 하기
model.save("./model/save_keras35_1.h5")
model.save(".//model/save_keras35_2.h5")
model.save(".\model\save_keras35_3.h5")
model.save(".\\model\\save_keras35_3.h5")
'''


'''
집에가서 왜 이렇게 나오는지 계산하기
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 200)               161600
_________________________________________________________________
dense (Dense)                (None, 100)               20100
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_2 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 187,991
Trainable params: 187,991
Non-trainable params: 0
_________________________________________________________________
'''