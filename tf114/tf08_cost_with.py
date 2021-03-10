import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]

w = tf.compat.v1.placeholder(tf.float32)

# tf.Variable을 언급을 안해주면 
# sess.run(tf.global_variables_initializer())을 안해도 된다

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y)) 
# cost(=loss)는 커스텀 가능하다 
# mse, mae는 abs넣기
w_history = []
cost_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i * 0.1 # 그래프 그리기 편하게 0.1 단위로 한것
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_history.append(curr_w) # w_history = [] 위 리스트에 하나씩 넣기
        cost_history.append(curr_cost)

print('==================================')
print(w_history)
print('================================')
print(cost_history)
print('================================')
 
plt.plot(w_history, cost_history)
plt.show()
# weight가 2일 때 loss가 가장 낮다



