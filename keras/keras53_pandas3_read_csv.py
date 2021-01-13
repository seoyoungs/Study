import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) 
#index_col인덱스 지정, header=0(디폴트 값), index_col의 디폴트는 none이다.
# header 2개면 1로 지정(0부터 시작하므로 2면 1이다.)

print(df)



