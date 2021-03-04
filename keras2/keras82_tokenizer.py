from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 먹었다'

token = Tokenizer() # 자연어 처리
token.fit_on_texts([text]) #  문자 데이터를 입력받아서 리스트의 형태로 변환

print(token.word_index)
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# 이렇게 '진짜' 처럼 중복되는 것이 젤 먼저 나오고 그 다음은 순서대로 인덱스 부여

x = token.texts_to_sequences([text]) # 시퀀스의 형태로 변환
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size) # 5(길이)
x = to_categorical(x) # 원핫 인코딩

print(x)
print(x.shape)
