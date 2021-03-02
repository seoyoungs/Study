import numpy as np

f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x - 4

x0 = 10.0 # 임의의 랜덤값
epoch = 60
learning_rate = 0.1 # learning_rate를 0.01로 하면 epoch를 300정도로 해야 미분한거랑 x 비슷하게 나옴

print('step\tx\tf(x)') # t : tab한번 (띄어쓰기)
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))
# 0 -> {:02d}, x0-> {:6.5f} , f(x0)-> {:6.5f}

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp

    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))

# 이차 함수 미분한 값 찾아내는 것







