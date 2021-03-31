# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

# 4번
# embadding

# 5번
# 시계열

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 자연어 처리의 전처리 단계 : 토큰화, 단어집합 생성, 정수인코딩, 패딩

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    with open('sarcasm.json') as file:
        data = json.load(file)

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    # 데이터를 sentence와 label로 나눈다
    for item in data:
        sentences.append(item['headline']) # 뉴스 기사의 헤드라인
        labels.append(item['is_sarcastic']) # 뉴스 헤드라인이 Sarcastic하다면 1, 그렇지 않다면 0.
    
    # 훈련 테스트와 테스트 데이터로 분
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok)
    # num_words 파라미터를 이용해서 단어의 개수를 제한, vocab_size 전체 단어 수
    # oov_token 인자를 사용하면 미리 인덱싱하지 않은 단어들은 ‘<OOV>’로 인덱싱
    tokenizer.fit_on_texts(training_sentences) 
    #fit_on_texts() 메서드는 문자 데이터를 입력받아서 리스트의 형태로 변환

    word_index = tokenizer.word_index # word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환
    
    training_squences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_squences, maxlen = max_length, 
                                     padding = padding_type, truncating=trunc_type)
    # pad_sequences : 이 시퀀스를 입력하면 숫자 0을 이용해서 같은 길이의 시퀀스로 변환

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                                  padding=padding_type, truncating=trunc_type)
    
    # =================== 모델링 ========================
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # 이진분류
    ])

    # ============ 훈련, 컴파일 ======================
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    model.fit(training_padded, training_labels, epochs = 30,
              validation_data = (testing_padded, testing_labels), verbose=1)

    # =============== 평가 ================
    acc = model.evaluate(testing_padded, testing_labels)[1]
    print('acc :', acc)
 
    return model
'''
acc : 0.8108013868331909
'''

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category4/mymodel.h5")


