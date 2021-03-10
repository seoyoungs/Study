import tensorflow as tf
tf.set_random_seed(66)

# 행렬
x_data = [[73, 51, 65], 
          [92, 98, 11], 
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y=data = [[152],
          [185],
          [180],
          [205],
          [142]]

# shape 행렬 단위로 하기
x = tf.placeholder(tf.float32, shape=[None,3]) # None,3 -> 행렬은 더 추가 가능
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name = 'weight') # y가 (5,1)이므로
b = tf.Variable(tf.random_normal([1]), name = 'bias') # bias는 1개

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b  # 행렬 메트릭스 연산은 matmul을 쓴다

# [실습] 만들기
cost = tf.reduce_mean(tf.square(hypothesis - y))
# Minimize. Need a very small learning rate for this data set
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
# train = optimizer.minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=0.5, 
                                epsilon=1e-06, use_locking=False, 
                                name='Adam')
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict = {x : x_data, y : data})
    if step % 10 == 0:
        print(step, 'cost:', cost_val, '\n예측값: ', hy_val)

