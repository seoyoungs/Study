
import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7] # list 형식

W = tf.Variable(tf.random_normal([3]), name = 'weight') # 기울기, random_normal(랜덤한 정규분포값 넣기)
b = tf.Variable(tf.random_normal([3]), name = 'bias') # 절편

sess = tf.Session()                           # 한 번만 실행 
sess.run(tf.global_variables_initializer())   # 한 번만 실행
print(sess.run(W), sess.run(b)) # [0.06524777 0.870543   0.68193936] [ 1.4264158  -0.09901392  0.3357661 ]

# --------------- y값 예측 -------------------
# 예측
hypothesis = x_train * W + b

# x_train은 placeholder로 넣을 것
# 츨력되는 것 hyper 

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) 
#(결과값 - 예측값의 평균 (mse))^2의 평균(1/n) -> cost비용 또는 손실
# loss값 최소만들기 위해 최적합 (loss = mse), grident
# cost(W) = 1/m*((m시그마i=1)*(Wx_i - y_i)^2)

# ---------- loss 최소화 optimizer --------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 갱신된 weight수식은 weight - learning_rate*gradient
train = optimizer.minimize(cost) # loss 최소값만들어 최적의 weight만든다
# optimizer에서 cost 실행

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 ==0: #epoch와 같음
        print(step, sess.run(cost), sess.run(W), sess.run(b)) # 차례대로 출력 (step 2000까지 나온다)

# cost 값 y, W : weight, b : bias
# 2000 1.0781078e-05 [1.9961864] [1.0086691]
# weight가 2에 가까워지고 bias가 1에 가까워진다


