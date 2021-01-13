#csv파일 numpy로 만들기
import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) 
# print(df)

# print(df.shape) #(150, 5)
# print(df.info()) #  target만 타입이 int64이다. 왜?

#pandas를 numpy로 변환
aaa= df.to_numpy()
print(aaa)
print(type(aaa)) # target값이 플로트로 바꼈다.(왜? numpy는 한가지 형식만돼서)

# bbb= df.values
# print(bbb)
# print(type(bbb)) ##aaa와 bbb둘중 하나 선택

np.save("../data/npy/iris_sklearn.npy", arr=aaa)


