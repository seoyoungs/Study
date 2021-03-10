import tensorflow as tf

sess = tf.Session()

# -------------------- variable ---------------------------------
x = tf.Variable([2], dtype = tf.float32, name = 'test') # 변수 2로 지정

init = tf.global_variables_initializer() # 변수, initailzer 초기화

sess.run(init) # 텐서 플로우에 쓸 수 있게 초기화

print(sess.run(x)) # [2.]
