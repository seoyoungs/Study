import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')

print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 변수초기화
aaa = sess.run(W)
print('aaa: ', aaa)
sess.close() # 종료

# sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer()) # 변수초기화
bbb = W.eval() # 위 aaa = sess.run(W) 이 방식과 같다
print('bbb: ', bbb)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session = sess)
print('ccc: ', ccc)
sess.close()




