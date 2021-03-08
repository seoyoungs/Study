# 이상치 처리
# 1. 0 처리
# 2. Nan처리 후 보간(bogan)
# outlier 처리법은 결측치와 비슷

# ------------------------------ 짝수 일 때 계산법 ------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75]) # 백분위수(Percentile)
#     print('1사분위 : ', quartile_1)
#     print('2사분위(중위값) : ', q2)
#     print('3사분위 : ', quartile_3)
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5) # -91
#     upper_bound = quartile_3 + (iqr * 1.5) # 191.75 이 두개 사이 이외에 있는 것을 이상치 
#     return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outlier_loc = outliers(aaa)
# print('이상치의 위치 : ', outlier_loc)

# # boxplot으로 이상치
# plt.boxplot(aaa)
# plt.show()

# ------------------------------ 홀수 일 때 계산법 ------------------------------------
import numpy as np
import matplotlib.pyplot as plt
aa1 = np.array([1,2,3,4,6,7,8,90,100,5000,10000])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75]) # 백분위수(Percentile)
    print('1사분위 : ', quartile_1)
    print('2사분위(중위값) : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5) # -91
    upper_bound = quartile_3 + (iqr * 1.5) # 191.75 이 두개 사이 이외에 있는 것을 이상치 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outliers(aa1)
print('이상치의 위치 : ', outlier_loc)

# boxplot으로 이상치
plt.boxplot(aa1)
plt.show()

# 홀수와 짝수일 때 이상치의 차이
# 짝수일때는 중간값 2개의 평균이지만
# 홀수는 중간값 하나를 찾는다

