#공부할 때 주석처리 풀면서 어떤 형식인지 공부하기
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

dataset = load_wine()
# x= dataset.data
x= dataset['data']  # 딕셔너리 형태로 바꿈
y= dataset['target']  # 딕셔너리 형태로 바꿈
# y= dataset.target
#df= pd.DataFrame(dataset.data) ## x
df = pd.DataFrame(x, columns=dataset.feature_names) # columns 데이터 이름
print(df.info())

df.columns=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

#x와 y붙이기
df['target']= y #dataset.target 이걸로도 y 대체 가능

df.to_csv('../data/csv/wine_sklearn.csv', sep=',') #csv파일 만들기
#sep=',' 구분은 , 로 하겠다.
