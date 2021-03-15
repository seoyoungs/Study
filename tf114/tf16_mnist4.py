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
# w1 = tf.Variable(tf.random_normal([784, 100]), name = 'weight1')
w1 = tf.get_variable('weight1', shape = [784, 100],
                     initializer = tf.contrib.layers.xavier_initializer())  # Variable의 다른 방법
b1 = tf.Variable(tf.random_normal([100]), name = 'bias1')
# layer1 = tf.nn.elu(tf.matmul(x, w) + b)      # 마지막꺼에만 softmax 넣기
# layer1 = tf.nn.selu(tf.matmul(x, w) + b)     # 마지막꺼에만 softmax 넣기
layer1 = tf.nn.selu(tf.matmul(x, w1) + b1)     # 마지막꺼에만 softmax 넣기
# layer1 = tf.nn.dropout(layer1, keep_prob=0.3)  # 30% 드랍아웃

# w2 = tf.Variable(tf.random_normal([100, 50]), name = 'weight2')
w2 = tf.get_variable('weight2', shape = [100, 64],
                      initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]), name = 'bias2')
layer2 = tf.nn.selu(tf.matmul(layer1, w2) + b2)
# layer2 = tf.nn.dropout(layer2, keep_prob=0.3) # 30% 드랍아웃

# w3 = tf.Variable(tf.random_normal([50, 10]), name = 'weight3')
w3 = tf.get_variable('weight3', shape = [64, 32],
                      initializer = tf.contrib.layers.xavier_initializer()) # xavier 말고 다른것 해보기
b3 = tf.Variable(tf.random_normal([32]), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.2)

# w4 = tf.Variable(tf.random_normal([64, 10]), name = 'bias4')
w4 = tf.get_variable('weight4', shape = [32, 10],
                      initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4) # 최종 아웃풋 softmax

# ------------ 3. 컴파일, 훈련 ------------------
# cost = tf.reduce_mean(tf.square(hypothesis - y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis= 1))   # 연산하면 마이너스로 수렴
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001).minimize(loss)   # 최소화 과정

training_epochs = 150
batch_size = 100
total_batch = int(len(x_train)/(batch_size))  # 60000 / 100 = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):   # 1_epoch = 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]   # batch_size 안에 들어간다
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict= feed_dict)
        avg_cost += c/total_batch  
        # 60000개를 100개 씩 나눠서 600번 돌려서 나온 것을 600으로 나눠서 평균 낸 것
        # 속도를 빠르게 하기 위해 한 것(batch_size가 너무 크면 메모리가 터질 수 있다)
        # 이렇게 batch_size를 다 지정해 줘야 한다.
    
    print('Epoch : ', '%04d' %(epoch + 1),
            'cost = {:.9f}'.format(avg_cost))

print('훈련 끝~!!!')

prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(prediction, dtype = tf.float32))
print('acc : ', sess.run(acc, feed_dict= {x:x_test, y:y_test}))

# acc :  0.5223




'''
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
    a = sess.run(hypothesis, feed_dict={x:x_test})
    print("acc: ",accuracy_score(sess.run(tf.argmax(y_test,1)),sess.run(tf.argmax(a,1))))
    # predict
    # pred = sess.run(hypothesis, feed_dict = {x:x_test})
    # pred = sess.run(tf.argmax(pred, 1))
    # print('pred  : ', pred[:7])
    # print('y_test:', y_test[:7])
    # #accuracy_score
    # acc = accuracy_score(pred, y_test)
    # print('acc : ', acc)
'''



