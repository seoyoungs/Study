#공부할 때 주석처리 풀면서 어떤 형식인지 공부하기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()


x= dataset.data
#x= dataset['data']  # 딕셔너리 형태로 바꿈
#y= dataset['target']  # 딕셔너리 형태로 바꿈
y= dataset.target
#df= pd.DataFrame(dataset.data) ## x
df = pd.DataFrame(x, columns=dataset.feature_names) # columns 데이터 이름

df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

#x와 y붙이기
df['target']= y #dataset.target 이걸로도 y 대체 가능

df.to_csv('../data/csv/iris_sklearn.csv', sep=',') #csv파일 만들기
#sep=',' 구분은 , 로 하겠다.




