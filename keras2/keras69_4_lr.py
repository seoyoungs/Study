x_train = 0.5 # n인 x를 여러개 하려면?
y_train = 0.8

#### 애네들 바꿔가면서 조절
weight = 0.5
lr = 0.01
epoch = 200

for iteration in range(epoch):
    y_predict = x_train * weight # 0.5*0.5
    error = (y_predict - y_train) **2 #(0.25-0.8)**2 = (0.55)**2

    print('Error : ' + str(error) + '\ty_predict : ' + str(y_predict)) # Error = loss

    up_y_predict = x_train*(weight + lr) # 0.5*0.51
    up_error = (y_train - up_y_predict)**2 # mse -> (1/n)(yi- y^)^2 : n의 개수가 하나이므로 뒤에것만

    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) **2 # mse -> (1/n)(yi- y^)^2

    if(down_error <= up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr

# loss가 점점 줄면서 y_predict가 0.8에 가까워진다
# (미분한것 * lr)
# Error : 1.9721522630525295e-31  y_predict : 0.8000000000000005 뒤에 5가 남는 것은
# float형식이라 계산을 못한다 그래서 5가 남음

