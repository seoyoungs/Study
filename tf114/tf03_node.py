# tf 114 에서 진행

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3)  
# 1,2 점대 텐서 나오는 것
 # tf.Tensor(7.0, shape=(), dtype=float32)
 # Tensor("Add:0", shape=(), dtype=float32)
 
sess = tf.Session()
print('sess.run(node1, node2) : ', sess.run([node1, node2]))
print('sess.run(node3) :', sess.run(node3))
# sess.run(node3) : 7.0
