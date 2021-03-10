from sklearn.datasets import load_boston, load_diabetes
from sklearn import datasets, metrics, preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


# [실습]

# 최종 sklearn의 r2 값으로 결론낼 것

boston = datasets.load_diabetes()
x_data = boston.data
y_data = boston.target

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data
                                    , train_size = 0.8, random_state = 66)
y_train = np.reshape(y_train,(-1,1))

# standardScaler = StandardScaler()
# standardScaler.fit(x_train)
# x_train = standardScaler.transform(x_train)
# x_test = standardScaler.transform(x_test)
# print(x_train.shape, y_train.shape) # (331, 10) (331, 1)

x = tf.placeholder(dtype=tf.float32,shape=[None,10])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([10,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis = tf.add(tf.matmul(data_input,a),b)
hypothesis = tf.matmul(x, w)+b

cost = tf.reduce_mean(tf.square(hypothesis - y))  # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.8)
train = optimizer.minimize(cost)
  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _,cost_val, h_val = sess.run([train, cost, hypothesis],
                                 feed_dict={x:x_train, y:y_train})
        
        y_pred = sess.run(hypothesis, feed_dict={x:x_test})
        
        if step % 1000 ==0 :
            print(step, cost_val, h_val)
    
    print("R2 : ", r2_score(y_test,y_pred))

# R2 :  0.5026611253458089

