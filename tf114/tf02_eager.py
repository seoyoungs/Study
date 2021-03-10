# 즉시 실행 모드
# --------- 2점대에도 1점대 텐서 실행하기 -----------------
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

# print(tf.executing_eagerly()) # False

# tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly()) # False

# ------------ 텐서 2점대로 다시하기

print(tf.executing_eagerly()) # True

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False
# ------- 텐서 플로2부터 즉시실행 코드가 없다. sess가 없어도 된다


# conda base에서는 (2점대에서는) session이 없다
# AttributeError: module 'tensorflow' has no attribute 'Session'

# disable_eager_execution 텐서플로에 2 점대에서 1점대 하려고 하면 요거 풀어야한

print(tf.__version__)

hello = tf.constant('hello World')
print(hello) # 텐서 1에서는 이렇게 하면 자료형만 출력된다
# Tensor("Const:0", shape=(), dtype=string)

# 텐서 1에서는 모든 자료가 그냥 print안되고
# 셰션이라는 것을 만들어 통과시켜야 한다

# sess = tf.Session()    # session 형성
sess = tf.compat.v1.Session() # 2점대 keras에서 이렇게실행시킹 구 있다
print(sess.run(hello)) # sess도 같이 입력해야지 원하는 글자가 나온다

'''
정리
1점대에서는 Session으로 실행해야한다
그런데 2점대에서도 1점대 실행가능하다
tf.compat.v1.Session 먼저 언급후 실행해야 한다
sess = tf.compat.v1.Session() # 2점대 keras에서 이렇게실행시킹 구 있다
print(sess.run(hello)) 
'''
