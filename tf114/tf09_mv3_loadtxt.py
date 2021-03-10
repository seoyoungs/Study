# 프레딕트 타입으로 하기
# float32 형태
# 73.,80.,75.,152.
# 93.,88.,93.,185.
# 89.,91.,90.,180.
# 96.,98.,100.,196.
# 73.,66.,70.,142.


import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

dataset = np.loadtxt('data-01-test-score.csv', 
                      delimiter=',', dtype = np.float32)

x = dataset[:5, :-1]
y = dataset[:5, [-1]] # 이렇게 하면 자동으로 (25,1)이 부여됨
# print(y)
# print(x.shape, y.shape) # (25, 3) (25,)

# ------------ placeholder 지정 -----------------------
x_train = tf.placeholder(tf.float32, shape=[None, 3])
y_train = tf.placeholder(tf.float32, shape=[None, 1])

# ------------- Variable(변수) 정의 --------------------
w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# ------------- hypothesis, cost 정의 -----------------
hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

# ------------- optimizer ------------------------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={x_train :x, y_train :y})
    if step%10 == 0:
        print(step, 'cost:', cost_val, '\n 예측값 \n:', hy_val)

# print("your score will be", 
#          sess.run(hypothesis, feed_dict={x_train: [[73.,80.,75.],
#                                                     [93.,88.,93.],
#                                                     [89.,91.,90.],
#                                                     [96.,98.,100.],
#                                                     [73.,66.,70.]], 
#                                         y_train: [[152],[185],[180],[196], [142]]}))

# cost: 6.502113 
'''
cost: 2.1434932 
 예측값
: [[151.41911]
 [184.45918]
 [180.29016]
 [198.2605 ]
 [139.78787]]
'''

