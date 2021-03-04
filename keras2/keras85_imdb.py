from tensorflow.keras.datasets import reuters, imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, Conv1D

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=5000
)

print('====================================')
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)

# 실습 embadding 으로 만들것
# ============ 전처리 ======================
x_train = pad_sequences(x_train, padding='pre', maxlen = 500)
x_test = pad_sequences(x_test, padding='pre', maxlen = 500)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ================= 모델링 ===========================
model = Sequential()
model.add(Embedding(input_dim = 5000, output_dim = 128, input_length=500))
# model.add(LSTM(units=32, dropout=0.3, recurrent_dropout=0.3))
model.add(Conv1D(filters=15, kernel_size=1))
model.add(Flatten())
model.add(Dense(50, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size = 32) #, callbacks =[es]

acc = model.evaluate(x_test, y_test)[1] # loss랑 acc중 2번째
print('acc: ', acc)


