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


print(x_train[0], type(x_train[0]))
print(y_train[0])
print(len(x_train[0]), len(x_train[11])) # 87 59
print('====================================')
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print('뉴스기사 최대길이 : ', max(len(l) for l in x_train)) # 뉴스기사 최대길이 :  2376
print('뉴스기사 평균길이 : ', sum(map(len, x_train))/ len(x_train)) # 뉴스기사 평균길이 :  145.5398574927633

# plt.hist([len(s) for s in x_train], bins = 40)
# plt.show() # x가 데이터 길이

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True) # 해석 써놓기
print('y분포 :', dict(zip(unique_elements, counts_elements)))
print('===============================================')

plt.hist(y_train, bins = 46)
plt.show() # y가 데이터 길이 # y는 이진분류

# x의 단어들 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index)) # 단어 유형 --> word_size = input_dim
print('----------------------------------------')

# 키와 밸류를 교체
index_to_word = {} # 딕셔너리 하나 생성
for key, value in word_to_index.items():
    index_to_word[value] = key

# 키 밸류 교환 후 
print(index_to_word) # 'mdbl': 10996,  --> 10996: 'mdbl'
print(index_to_word[1]) # the 가장 많이 썼다는 뜻
print(len(index_to_word)) # 30979
print(index_to_word[30979])

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

'''
# ================= 전처리 ==============================
x_train = pad_sequences(x_train, padding='pre', maxlen = 500)
x_test = pad_sequences(x_test, padding='pre', maxlen = 500)
y_train = to_categorical(y_train)        # y도 원핫인코딩 꼭 하기
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
print(y_train.shape, y_test.shape) # (8982, 100) (2246, 100)
# ================= 모델링 ===========================
model = Sequential()
model.add(Embedding(input_dim = 5000, output_dim = 128, input_length=500))
# model.add(LSTM(units=32, dropout=0.3, recurrent_dropout=0.3))
model.add(Conv1D(filters=15, kernel_size=1))
model.add(Flatten())
model.add(Dense(50, activation='tanh'))
model.add(Dense(46, activation='softmax'))
model.summary()
# es = EarlyStopping(monitor= 'val_loss', mode= 'min', verbose = 5, patience= 10)
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size = 32) #, callbacks =[es]
acc = model.evaluate(x_test, y_test)[1] # loss랑 acc중 2번째
print('acc: ', acc)
'''


