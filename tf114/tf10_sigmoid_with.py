import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2], [2,3], [3,1],
          [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0],
          [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# ------------- Variable(변수) 정의 --------------------
w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # sigmoid가 0,1 사이로

# cost = tf.reduce_mean(tf.square(hypothesis - y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

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






