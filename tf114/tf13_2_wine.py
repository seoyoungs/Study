from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
from sklearn import datasets, metrics, preprocessing
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

dataset = load_wine()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)

dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, random_state=44
)

# ------------------------ 전처리 --------------------
# #품종 column을 one-hot-encode
oh_encoder = OneHotEncoder(categories='auto')
oh_encoder.fit(y_train)
y_train = oh_encoder.transform(y_train).toarray()

# random_normal과 zeros 같이 쓰지 않기!
# minMaxScaler = MinMaxScaler()
# minMaxScaler.fit(x_train)
# x_train = minMaxScaler.transform(x_train)
# x_test = minMaxScaler.transform(x_test)
# print(x_train.shape, y_train.shape) # (142, 13) (142, 3)

x= tf.placeholder(tf.float32, shape=[None,13])
y= tf.placeholder(tf.float32, shape=[None,3])

w = tf.Variable(tf.zeros([13, 3]),name='weight')
b = tf.Variable(tf.zeros([1, 3]), name='bias')

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
train = tf.train.GradientDescentOptimizer(learning_rate= 1e-5).minimize(loss)   # 최소화 과정

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
    for step in range(5001):    
        cost_val, _ = sess.run([loss, train], feed_dict={x:x_train, y:y_train})

        if step % 200 == 0 :    
            print(step, cost_val)

    # h,c,a = sess.run([hypothesis,predicted, acc], 
    #                   feed_dict = {x: x_test, y: y_test})
    # print('예측값 : ', h, '\n 원래값 : ', c, 'acc: ', a)
    
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print("accuracy_score: ", accuracy_score(y_test, y_pred))

# -------------------------------------------------------------
# acc:  0.8240741
# accuracy_score:  0.6944444444444444


