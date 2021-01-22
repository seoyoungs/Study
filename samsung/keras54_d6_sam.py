import numpy as np
import pandas as pd

p_data = pd.read_csv(r'C:\data\csv\samsung1.csv', encoding='cp949', index_col=0, header=0)
data = pd.read_csv(r'C:\data\csv\samsung2.csv', encoding='cp949', index_col=0, header=0)
p_data = pd.concat([data,p_data])
# print(p_data)  

p_data_x = p_data.loc['2018-05-04':'2021-01-13']
p_data_y = p_data.loc['2018-05-08':'2021-01-14'] #하루 차이 나게 만들기

p_data_x = p_data.iloc[:,[0,1,2,5,6]]
p_data_y = p_data.iloc[:,[3]]

s_x = p_data_x.to_numpy()
s_y = p_data_y.to_numpy()

np.savez('../data/npy/s1_data.npz', x=s_x, y=s_y)


