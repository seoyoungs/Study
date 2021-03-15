# [실습]
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --------------- 전처리 ------------------------
x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()
print(y_train.shape, y_test.shape)

# ------------------ 모델링 -----------------------
# input place holders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
'''
# weights & bias for nn layers
w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
'''
w1 = tf.Variable(tf.random_normal([784,10],stddev=0.1),name = 'weight1')
b1 = tf.Variable(tf.random_normal([10],stddev=0.1),name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([10,10],stddev=0.1),name = 'weight2')
b2 = tf.Variable(tf.random_normal([10],stddev=0.1),name = 'bias2')

hypothesis = tf.nn.softmax(tf.matmul(layer1, w2) + b2)

# cost = tf.reduce_mean(tf.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis= 1))   # 연산하면 마이너스로 수렴
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)   # 최소화 과정
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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

    # pred = sess.run(hypothesis, feed_dict = {x:x_test})
    # pred = sess.run(tf.argmax(pred, 1))
    # print('pred  : ', pred[:7])
    # print('y_test:', y_test[:7])

    # #accuracy_score
    # acc = accuracy_score(pred, y_test)
    # print('acc : ', acc)



'''
0 15.482827
200 4.942831
400 2.2123837
600 1.4343623
800 1.1020312
1000 0.91573066
acc:  0.8038
다층 레이어
0 2.354815
200 0.5034779
400 0.3207165
600 0.27830052
800 0.25733438
1000 0.24329089
acc:  0.927
'''

