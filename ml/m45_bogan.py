# bogan은 시계열 데이터에서 많이 쓴다
# 선형이고 시계열일때 잘 먹는다
# 결측치 값(NAN)을 잡는것
# bogan 결측치에 predict해서 넣는것 

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print('===============================================')

ts = Series([1, np.nan, np.nan, 8, 10], index = dates) # Series 일때를 나타냄
# np.nan 결측값
print(ts)
# 2021-03-01     1.0
# 2021-03-02     NaN
# 2021-03-03     NaN
# 2021-03-04     8.0
# 2021-03-05    10.0
# dtype: float64

ts_intp_linear = ts.interpolate() # 보강법 (pandas에서 제공--> 시계열데이터에 조음)
print(ts_intp_linear)
# 2021-03-01     1.000000
# 2021-03-02     3.333333
# 2021-03-03     5.666667
# 2021-03-04     8.000000
# 2021-03-05    10.000000
# dtype: float64