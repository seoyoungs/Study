import tensorflow as tf
print(tf.__version__)

hello = tf.constant('hello World')
print(hello) # 텐서 1에서는 이렇게 하면 자료형만 출력된다
# Tensor("Const:0", shape=(), dtype=string)

# 텐서 1에서는 모든 자료가 그냥 print안되고
# 셰션이라는 것을 만들어 통과시켜야 한다

sess = tf.Session()    # session 형성
print(sess.run(hello)) # sess도 같이 입력해야지 원하는 글자가 나온다

