
# 최종 sklearn의 accracy_score 값으로 결론낼 것
# # 이진분류

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from sklearn import datasets, metrics, preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)










tf.set_random_seed(66)

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)
# (569, 30)
# (569, 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

x= tf.placeholder(tf.float32, shape=[None,30])
y= tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  #binary_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 

#텐서를 새로운 형태로 캐스팅하는데 사용한다.
#부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
#Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
#0.5 이상 1 0.5 보다 작으면 0

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val,hy_val ,_ = sess.run([cost, hypothesis,train], feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(f'step : {step} \ncost : {cost_val} \nhy_val : \n{hy_val}')


    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test, y:y_test})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')












'''

boston = datasets.load_breast_cancer()
x_data = boston.data
y_data = boston.target
y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data
                 , train_size = 0.8, random_state = 66)


# standardScaler = StandardScaler()
# standardScaler.fit(x_train)
# x_train = standardScaler.transform(x_train)
# x_test = standardScaler.transform(x_test)
# print(x_train.shape, y_train.shape) # (426, 30) (426, 1)

x = tf.placeholder(dtype=tf.float32,shape=[None,30])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])

w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')


hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=1e-05)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
#부동소수점형에서 정수형으로 바꾼 경우 소수점 버림을 한다.
#Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
#0.5 이상 1 0.5 보다 작으면 0

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val,hy_val ,_ = sess.run([cost, hypothesis,train], feed_dict={x:x_train, y:y_train})

        if step % 100 == 0:
            print(f'step : {step} \ncost : {cost_val} \nhy_val : \n{hy_val}')


    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test, y:y_test})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')

# accuracy: : 0.359649121761322
'''







