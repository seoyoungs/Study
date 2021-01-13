#공부할 때 주석처리 풀면서 어떤 형식인지 공부하기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
# print(dataset.keys())
#dict_keys(['data'(x값), 'target(y값)', 'frame', 'target_names', 
#          'DESCR', 'feature_names', 'filename(파일경로)'])
# print(dataset.values())
# print(dataset.target_names) #['setosa' 'versicolor' 'virginica']

x= dataset.data
#x= dataset['data']  # 딕셔너리 형태로 바꿈
#y= dataset['target']  # 딕셔너리 형태로 바꿈
y= dataset.target
#df= pd.DataFrame(dataset.data) ## x

# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)
# print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
###numpy를 pandas로 바꿀것

df = pd.DataFrame(x, columns=dataset.feature_names) # columns 데이터 이름
#df = pd.DataFrame(x, columns=dataset['feature_names']) # 딕셔너리 형태로 바꿈
# print(df) #header는 data아니므로 columns으로 해야함, 출력시 shape도 같이 보여줌
# #hearder, index는 데이터가 아니다. 그래서 작업 할 때 numpy 바꿔야함
# print(df.shape) #list에서는shape 안됨 (150, 4)
# print(df.columns)
# print(df.index)

# print(df.head()) #df[:5]
# print(df.tail()) #df[-5:] #-5부터끝
# print(df.info()) #non null 비어있는 값이 없다. 
# print(df.describe())

df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# print(df.columns)
# print(df.info()) #non null 비어있는 값이 없다. 
# print(df.describe())

#x와 y붙이기
print(df['sepal_length'])
df['target']= dataset.target
print(df.head()) #앞에서 5개

# print(df.shape) #(150,5)
# print(df.columns)
# print(df.index)
# print(df.tail())
# print(df.info())
# print(df.isnull()) # 결측치 있나 확인
# print(df.isnull().sum()) #0이면 결측치 없음
# print(df.describe())

# 상관계수 히트맵
# print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, 
#              annot=True, cbar=True) # heatmap 사각형 형태로, annot:글자넣기
# plt.show()

##도수 분포도(히스토그램) --전에 fit에서한 history는 loss와 metircs값
print(df['target'].value_counts())
# 2    50
# 1    50
# 0    50

plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')
plt.grid() #이거 격자 모양 추가 하고 싶으면 하기

plt.subplot(2,2,2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')
plt.grid()

plt.subplot(2,2,3)
plt.hist(x='petal_length', data=df, color='orange')
plt.title('petal_length')
plt.grid()

plt.subplot(2,2,4)
plt.hist(x='petal_width', data=df, color='red')
plt.title('petal_width')
plt.grid()

plt.show()

