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
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def solution_model():
    # 데이터
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # 모델 - relu 말고, linear사용
    model = Sequential()
    model.add(Dense(128, input_dim =1))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(4))
    model.add(Dense(1, activation='linear')) # 마지막은 linear

    # 컴파일, 훈련
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])
    model.fit(xs, ys, batch_size=1, epochs=200)

    # 평가, 예측 -->model.predict([10.0])을 넣은값 출력
    loss = model.evaluate(xs, ys)
    y_pred = model.predict([10.0])
    print(y_pred) # 11이 나와야 좋은 모델

    return model
'''
[[11.003363]]
'''

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category1/mymodel.h5") # 저장하면 모델이 여기에 생긴다
