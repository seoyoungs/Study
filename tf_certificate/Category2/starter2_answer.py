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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
# ==========================================================================
# # Fashion MNIST 데이터 셋에 대한 분류기 생성
# 테스트는 10 개의 클래스를 분류 할 것으로 예상하고
# 입력 모양은 Fashion MNIST 데이터 세트의 기본 크기 여야합니다.
# 28x28 단색. 데이터 크기를 조정하지 마십시오. 입력 레이어는
# (28,28)을 입력 모양으로 만 사용합니다. 이를 수정하면 테스트가 실패합니다.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

#conv1d나 Dense로 사용 shape가 (28, 28)이므로 
def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape) # (60000,) (10000,)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                      train_size = 0.8, shuffle=True, random_state = 6)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # 모양은 원래 (60000, 28, 28)이므로 전처리만
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_val = x_val.astype('float32')/255.

    # ========= 모델 ==============
    model = Sequential()
    # model.add(Conv1D(filters=50, kernel_size=2, padding='same', input_shape = (28, 28)))
    model.add(Dense(56, input_shape = (28, 28)))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(10, activation= 'softmax'))
    
    # ============= 컴파일, 훈련 ==============
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=50)

    # ============= 평가, 예측 ================
    acc = model.evaluate(x_test, y_test, batch_size=32)
    print('acc', acc[1])
    # y_pred = model.predict(x_test)
    # print(y_pred)

    # YOUR CODE HERE
    return model

'''
acc 0.8335000276565552
'''

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category2/mymodel.h5")




