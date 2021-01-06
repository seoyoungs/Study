import numpy as np

a = np.array(range(1,11)) #1부터 10까지
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] #위치 잘 맞춰서 넣어주기
        aaa.append(subset)
        #aaa.append([item for item  in subset])
    print(type(aaa))
    return np.array(aaa)
 
dataset = split_x(a, size)
print('------------------')
print(dataset)


