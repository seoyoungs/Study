# 최종 sklearn의 r2 값으로 결론낼 것

from sklearn.datasets import load_boston
from sklearn import datasets, metrics, preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

dataset = load_boston()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, random_state = 66)

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(x_train)
x_train = minMaxScaler.transform(x_train)
x_test = minMaxScaler.transform(x_test)
print(x_train.shape, y_train.shape) # (379, 13) (379, 1)

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

x_in = tf.placeholder(tf.float32,shape=[None,13])
y_in = tf.placeholder(tf.float32, shape=[None,1])

# hypothesis = tf.add(tf.matmul(data_input,a),b)
hypothesis = tf.matmul(x_in, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_in))  # MSE
train = tf.train.AdamOptimizer(learning_rate=0.8).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _,cost_val, h_val = sess.run([train, cost, hypothesis],feed_dict={x_in:x_train, y_in:y_train})
        
        y_pred = sess.run(hypothesis, feed_dict={x_in:x_test})
        
        if step % 1000 ==0 :
            print(step, cost_val, h_val)
    
    print("R2 : ", r2_score(y_test,y_pred))



# R2 :  0.8298558195812753





