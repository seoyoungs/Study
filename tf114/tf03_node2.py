##[실습]
# 텐서 1로
#  덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 나머지
# 만들기

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# ------------- 덧셈 ---------------------------
node3 = tf.add(node1, node2)

print(node3)  
# 1,2 점대 텐서 나오는 것
 # tf.Tensor(7.0, shape=(), dtype=float32)
 # Tensor("Add:0", shape=(), dtype=float32) 
sess = tf.Session() # 요거는 한 번만 정의하기
print('sess.run(node1, node2) : ', sess.run([node1, node2]))
print('sess.run(node3) :', sess.run(node3))
# sess.run(node3) : 5.0

# -------------- 뺄셈 ---------------------------

node4 = tf.subtract(node1, node2)
# sess = tf.Session()
print('sess.run(node4) :', sess.run(node4)) # sess.run(node4) : -1.0

# --------------- 곱셈 ------------------------
node5 = tf.multiply(node1, node2) # 원소곱, matmul 행렬곱
# sess = tf.Session()
print('sess.run(node5) :', sess.run(node5)) # sess.run(node5) : 6.0

# -------------- 나눗셈 -------------------------
node6 = (tf.divide(node1, node2))
print('sess.run(node6) :', sess.run(node6)) # sess.run(node6) : 0.6666667

# -------------- 나머지 --------------------------
node7 = (tf.mod(node1, node2))
print('sess.run(node7) :', sess.run(node7)) # sess.run(node7) : 2.0
