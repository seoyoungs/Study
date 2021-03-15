import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],     # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],     # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]    # 0


x = tf.placeholder('float', shape = [None, 4])     # 행은 늘릴 수 있다(데이터 양 증가)
y = tf.placeholder('float', shape = [None, 3])     # softmax 적용해야함

# ------------------- layer 구성 ----------------------------
w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')    # x,y의 열을 행렬로
b = tf.Variable(tf.random_normal([1, 3]))   
# weight 1개 당 bias는 통상 원래 1개이다(그러므로 None, 3이므로 [1,3]이다)

# ---------------- hypothesis softmax -------------------------
# softmax특징 모두 더하면 1이다
# 모델 통과 할 때마다 activation을 감싼다(여기서는 softmax)
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# ([None, 3] + [1, 3] = [None, 3])
# cost = 1/m(h - y)^2

# --------------- cost(loss) 구하기 --------------------------
# cost = tf.reduce_mean(tf.square(hypothesis - y))  # mse
# crossecentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis= 1))   # 연산하면 마이너스로 수렴
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)   # 최소화 과정

# ------------ for문 ----------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # predict
    a = sess.run(hypothesis, feed_dict = {x:[[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a,1))) # argmax : 가장 높은 인덱스한테 1 부여
    # [0] --> 첫번째가 가장 크다
# -------------------------------------------------------------

