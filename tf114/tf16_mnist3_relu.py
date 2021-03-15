import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# ----------- 1. 데이터 ---------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

# -------------------- 2. 모델 구성 ---------------------
# ----------------------------------------------- 레이어 구성
w = tf.Variable(tf.random_normal([784, 100]), name = 'weight1')
b = tf.Variable(tf.random_normal([100]), name = 'bias1')
# layer1 = tf.nn.elu(tf.matmul(x, w) + b) # 마지막꺼에만 softmax 넣기
# layer1 = tf.nn.selu(tf.matmul(x, w) + b) # 마지막꺼에만 softmax 넣기
layer1 = tf.nn.relu(tf.matmul(x, w) + b) # 마지막꺼에만 softmax 넣기
layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # 30% 드랍아웃

w2 = tf.Variable(tf.random_normal([100, 50]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([50]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3) # 30% 드랍아웃

w3 = tf.Variable(tf.random_normal([50, 10]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3) # 최종 아웃풋 softmax

# ------------ 3. 컴파일, 훈련 ------------------
# cost = tf.reduce_mean(tf.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis= 1))   # 연산하면 마이너스로 수렴
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)   # 최소화 과정
train = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

# ------------ 예측값 ----------------------------------------
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
#부동소수점형에서 정수형으로 바꾼 경우 소수점 버림을 한다.
#Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
#0.5 이상 1 0.5 보다 작으면 0

acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

# ------------- for문 ----------------------------------------
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    for step in range(1001):    
        cost_val, _ = sess.run([loss, train], feed_dict={x:x_train, y:y_train})

        if step % 200 == 0 :    
            print(step, cost_val)

    # a = sess.run(hypothesis, feed_dict={x:x_test})
    # print("acc: ",accuracy_score(sess.run(tf.argmax(y_test,1)),sess.run(tf.argmax(a,1))))
    # predict
    pred = sess.run(hypothesis, feed_dict = {x:x_test})
    pred = sess.run(tf.argmax(pred, 1))
    print('pred  : ', pred[:7])
    print('y_test:', y_test[:7])

    #accuracy_score
    acc = accuracy_score(pred, y_test)
    print('acc : ', acc)
