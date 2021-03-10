# tf 114 에서 진행
# placeholder

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # 상수 값을 사용하기 위해서는 tf.constant() 함수를 사용해서 상수 x 값을 생성하면 됩니다
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]})) # feed_dict 딕셔너리 형태로 넣어라

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2})) # 6*3 =18


# a,b라는 node를 input하고 add를 해 session으로 노출시킨다
# 숫자말고 리스트 형태로도 넣을 수 있다
# placeholder, Session, variable 계산하는 것이 아니라 입력 값을 받는 것

