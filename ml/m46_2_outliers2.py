# 실습
# outliers1을 행렬형태로도 적용할 수 있도록 수정

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# 이렇게 행렬 이상치 위치를 각각 구하는 이유
# 한개의 행렬은 태양열, 한개는 주식일 수 있으므로
# 연관이 없는 것은 이렇게 각각 구해야한다

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100], 
              [100, 200, 3, 400, 500, 600, 700, 8, 900, 1000]]) #shape (2, 10)
aaa = aaa.transpose()
# aaa = pd.DataFrame(data=aaa)
print(aaa.shape) # (10,2) 컬럼을 퀀타일

# --------------- 행렬 이상치 위치 각각 구하는 함수 ----------------------
list = []
def outliers(data_out):
    for col in range(data_out.shape[1]): # 컬럼을 사분위수 하므로 [1]로 지정
        # print(data_out[:,col])
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,col], [25, 50, 75]) # 백분위수(Percentile)
        print('1사분위 : ', quartile_1)
        print('2사분위(중위값) : ', q2)
        print('3사분위 : ', quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5) # -91
        upper_bound = quartile_3 + (iqr * 1.5) # 191.75 이 두개 사이 이외에 있는 것을 이상치 
        outlier_loc = np.where((data_out[:,col]>upper_bound) | (data_out[:,col]<lower_bound))
        list.append(outlier_loc)
    return np.array(list)

outlier_loc = outliers(aaa)
print('이상치의 위치 : ', list)
# 이상치의 위치 :  [(array([4, 7], dtype=int64),), (array([], dtype=int64),)]
# 마지막은 이상치 범위가 넓어서 이상치가 없는 것으로 나온다

plt.boxplot(aaa)
plt.show()
