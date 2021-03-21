import tensorflow as tf
import numpy as np
# tf.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly()) 
# print(tf.__version__)

# tf114말고 base에서 하기
# AttributeError: module 'tensorflow' has no attribute 'placeholder'
# -------- 에러뜬다 ---------------------

# ------------- gpu 2개로 돌리는 방법 ------------------------
# gpu[1]로해서 이파일을 돌리고
# gpu[0]으로 해서 keras68_mnist2_gpu.py을 돌리면
# 한 번에 두개이상을 돌릴 수 있다
# 단, 글카 2개 ㅎㅎ 나만 된다 헤헿
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)


# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15 # 반복수
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000 / 100

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# ===================== 2. 모델구성 ===========================

# --------------- L1
# shape=[3, 3, 1, 32] 채널, (3,3,1)이 커널 사이즈, 32가 필터, 아웃풋
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 64])  
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') 
# strides 동일하게, 띄어쓰기 변경하려면 strides=[1, 1, 1, 1] 에서 두번째만 건들기
# Conv2D(filter, kernel_size, input_shape)   # 서머리?
# w1 = Conv2D(64, (3,3), input_shape = (28,28,1))   # 파라미터 개수? 70개
# 커널 사이즈 곱하기 채널 * 인풋 딤 
# 다음 파라미터는 28, 28, 32로 간다,  padding='SAME'으로 32다
print(L1)
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1) # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# -------------- L2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 64]) # 첫번째 아웃풋이 두번째 인풋  
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME') 
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)

# --------------- L3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128]) # 첫번째 아웃풋이 두번째 인풋  
L3 = tf.nn.conv2d(L2, w3, strides=[1, 1, 1, 1], padding='SAME') 
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)

# --------------- L4
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 128, 64]) # 첫번째 아웃풋이 두번째 인풋  
L4 = tf.nn.conv2d(L3, w4, strides=[1, 1, 1, 1], padding='SAME') 
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)
# shape가 1/2씩 줄어드는 이유?
# pool_size : 수직, 수평 축소 비율을 지정. (2,2)라면 출력 영상 크기는 입력 영상 크기의 반으로 줄어든다.

# --------------- Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64])
print('플래튼 : ',L_flat)

# --------------- L5.
w5 = tf.compat.v1.get_variable('w5', shape = [2*2*64, 64])
                    #  initializer = tf.contrib.keras.initializers.he_normal()) # he_initializer
b5 = tf.Variable(tf.compat.v1.random_normal([64], name = 'b5'))
L5 = tf.nn.selu(tf.matmul(L_flat, w5)+ b5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)

# --------------- L6
w6 = tf.compat.v1.get_variable('w6', shape = [64, 32])
                    #  initializer = tf.contrib.keras.initializers.he_normal()) # he_initializer
b6 = tf.Variable(tf.compat.v1.random_normal([32], name = 'b6'))
L6 = tf.nn.selu(tf.matmul(L5, w6)+ b6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)

# --------------- L7(출력 레이어)
w7 = tf.compat.v1.get_variable('w7', shape = [32, 10])
                    #  initializer = tf.contrib.layers.xavier_initializer()) # he_initializer
b7 = tf.Variable(tf.compat.v1.random_normal([10], name = 'b7'))
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+ b7)    # 마지막은 softmax
print('최종출력 : ', hypothesis)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # categorical_corossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

# 4. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

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
            'loss = {:.9f}'.format(avg_cost))

print('훈련 끝~!!!')

prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(prediction, dtype = tf.float32))
print('acc : ', sess.run(acc, feed_dict= {x:x_test, y:y_test}))


