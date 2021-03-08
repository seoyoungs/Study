x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]
# 파일 형태 list

print(x, '\n', y)

import matplotlib.pyplot as plt
plt.plot(x,y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y}) # list를 pandas로 역음
print(df)
print(df.shape) #(10, 2)
#ket value 형태로 컬럼명을 지정

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape) # (10,), (10,) 벡터(10,) -> 행렬(10,3) -> 텐서 (10,3,2)
print(type(x_train)) # <class 'pandas.core.series.Series'> -> 1차원이므로
# 2차원은 type가 dataframe형태로 나온다

x_train = x_train.values.reshape(len(x_train), 1)
print(x_train.shape, y_train.shape) # (10, 1) (10,)

from sklearn.linear_model import LinearRegression # 선형모델
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print('score : ', score)

print('기울기 : ', model.coef_) # weight
print('절편', model.intercept_) #bias
# # weight와 bias 평가 방법
# x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
# y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]
# 기울기 :  [2.]
# 절편 3.0
# 2배 곱해서 3을 더한것 이므로
