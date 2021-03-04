from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', 
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요', '별로예요',
        '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', 
        '참 재밌네요', '규현이가 잘생길 뻔 했어요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) # {'참': 1,}---> 원래는 정수 1부터 부여한다
x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # 나의 앞쪽이 다 0으로 채워짐, maxlen 파라미터 디폴트
# padding='pre'라 maxlen=4로 하면 4열까지만 되는데 pre라 앞에서 잘린다
#  [11 12 13 14 15] ->  [12 13 14 15]
print(pad_x) 
print(pad_x.shape)  # (13, 5)

print(np.unique(pad_x))
#[ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28] -> padding하면서 0이 들어간다
# 이렇게 위에 11이 없어진것을 볼 수 있다
print(len(np.unique(pad_x))) # 0 ~ 28까지면 개수는 29개인데 maxlen=4로 11이 없으므로 28개이다.

pad_x= pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

# ============ 임베딩 레이어 빼고 모델 구성 =============================
              # LSTM, Danse로 구성 가능 #
# 변수별 거리 계산을 수치화로 표시 => 백터화
model = Sequential()
# model.add(Embedding(input_dim = 29, output_dim=11, input_length=5)) # 지금 3차원 ---> PCA와 같은역할
                   # 백만개의 자료를 다 원핫인코딩 할 수 없으니까 
                   # Embedding으로 수치화(백터화)하고 자른다
                   # 만약 전체단어가 29개인데 작게 하면 돌아가나? input_dim을 작게 
                   # --> 안돌아간다, 크게 하면 -> 돌아간다

# (13, 5)이므로 input_length=5, input_dim = 29 x의 총 개수, output_dim(내가원하는 아웃풋의 dim)
# model.add(Embedding(29, 11))

# model.add(LSTM(32, activation='relu', input_shape=(pad_x.shape[1:]))) # 지금 3차원이므로 LSTM 그대로
model.add(Dense(32, input_shape=(pad_x.shape[1:])))
# model.add(Flatten()) 
                   # Embedding(input_dim = 29, output_dim=11, input_length=5)일때 flatten하면
                   # 3 -> 2차원으로 바뀐다 (None, 5, 11) 이므로 
                   # 근데 model.add(Embedding(29, 11))이면 (None, None, 11) 이므로 안된다
# model.add(Conv1D(32,2))
                    # 여기서 32,2에서 2는 커널을 2개씩 자른다는 뜻
                    # 왜 Conv1D되닝(3차원에서 출력값 Dense가 받아들인다)
                    # 정확한 측정을 위해서는 Flatten해주는 것이 좋다
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1] # loss랑 acc중 2번째
print('acc: ', acc)