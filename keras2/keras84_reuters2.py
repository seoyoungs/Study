# 단어 input_dim = 10000, maxlen 자르는 방법, 임베딩사용해서 모델링
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 5000, test_split=0.2
) # num_words = 10000, 10000번째 안에 있는 것을 가져온다

# =========== y 카테고리 갯수 출력
category = np.max(y_train) + 1
print('y 카테고리 개수 :', category)

# =========== y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

# ======================== 전처리 =============================
x_train = pad_sequences(x_train, padding='pre', maxlen = 500)
x_test = pad_sequences(x_test, padding='pre', maxlen = 500)
print(x_train.shape, x_test.shape) #(8982, 500) (2246, 500)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, Flatten

model = Sequential()
# model.add(Embedding(input_dim=5000, output_dim=128, input_dim=100))
model.add(Embedding(5000, 128)) # 이것도 가능
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
# 'sparse_categorical_crossentropy' 요거는 to_categorical 지우고 실행
model.fit(x_train, y_train, epochs=10, batch_size = 32, verbose=1)

result = model.evaluate(x_test, y_test) # loss랑 acc중 2번째
print('result: ', result)

'''
acc:  0.8288400173187256
'''
