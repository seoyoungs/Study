# tf10_sigmoid 가져와서 수정
# 예측값 1나오게 하기

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32) # 이진분류

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# ------------- Variable(변수) 정의 --------------------
# w = tf.Variable(tf.random_normal([2,1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')
# hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # sigmoid가 0,1 사이로

# ----------- acc: 1.0 나오게 하려면 w,b 레이어 층을 여러개 분배해야함 -----------------------
# ---------------- input layer
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1') # weight의 크기 2x2
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, W1) + b1)
# model.add(Dense(10, input_dim = 2, activation = sigmoid))

# ---------------- hidden layer (shape 자유롭다)
W2 = tf.Variable(tf.random_normal([10, 7]), name='weight2')
b2 = tf.Variable(tf.random_normal([7]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2) # layer1 = x
# model.add(Dense(7, activation = 'sigmoid'))

# ---------------- output layer
W3 = tf.Variable(tf.random_normal([7, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3) # layer2 = x
# model.add(Dense(1, activation = 'sigmoid'))

# cost = tf.reduce_mean(tf.square(hypothesis - y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
#부동소수점형에서 정수형으로 바꾼 경우 소수점 버림을 한다.
#Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
#0.5 이상 1 0.5 보다 작으면 0

acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))


with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    for step in range(5001):    
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0 :    
            print(step, cost_val)

    h,c,a = sess.run([hypothesis,predicted, acc], 
                      feed_dict = {x: x_data, y: y_data})
    print('예측값 : ', h, '\n 원래값 : ', c, 'acc: ', a)


