import numpy as np
dataset1= np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset1)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
           break
        tmp_x, tmp_y = dataset[i : end_number],  dataset1[end_number]
        x.append(tmp_x) #i가 0일 때 tmp_x는 dataset[0:4]이므로 [1,2,3,4]가 되고
        y.append(tmp_y) #, tmp_y는 dataset[4]이므로 다섯 번째의 숫자인 5
    return np.array(x), np.array(y)

x, y = split_xy1(dataset1, 3) #데이터 x를 4일치씩 나눈다.
print(x, "\n", y)


import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy2(dataset, x_column, y_column):     #y칼럼 매개변수 추가
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_column
        y_end_number = x_end_number + y_column       #추가
        # if end number > len(dataset) -1
        # break
        if y_end_number > len(dataset):              #수정
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number :y_end_number]  # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_column = 3
y_column = 2
x, y = split_xy2(dataset, x_column, y_column)
print(x, '\n', y)
print('x.shape : ', x.shape)
print('y.shape:', y.shape)

'''
[[1 2 3]
 [2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]
 [6 7 8]
 [7 8 9]] 
 [ 4  5  6  7  8  9 10]
[[1 2 3]
 [2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]
 [6 7 8]] 
 [[ 4  5]
 [ 5  6]
 [ 6  7]
 [ 7  8]
 [ 8  9]
 [ 9 10]]
x.shape :  (6, 3)
y.shape: (6, 2)
'''
