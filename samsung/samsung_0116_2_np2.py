import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('C:\data\csv\kodex2.csv', encoding='cp949', index_col=0, header=0, thousands=',')
df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
# df1.replace(',','',inplace=True, regex=True)
df1.replace(',','',inplace=True, regex=True)
df1 = df1.astype('float64') #실수화
df1 = df1.sort_values(by=['일자'], axis=0) #sort_values 일자로 정렬하기
df1=df1[424:]
print(df1)
df_x = df1.iloc[:-1,[1,2,3,5,6,8]]
df_y = df1.iloc[1:,[0]]
df_y.loc['2021-01-15'] = [4420]
print(df_x.shape)
df_x = df_x.to_numpy()
df_y = df_y.to_numpy()

# data2 = df1.to_numpy

# df1_x = df1.loc['2018/05/04': '2021/01/14']
# df1_y = df1.loc['2018/05/08' : '2021/01/15']
# # df2_x = df1.loc['2021/01/15':'2021/01/15' ]
# print(df1_x.shape)
# print(df1_y.shape)

# df1_x = df1_x.iloc[:,[1,2,3,5,6,12]]
# df2_x = df2_x.iloc[:,[1,2,3,5,6,12]]
# df1_y = df1_y.iloc[:,[0]]
# # df_y.loc['2021-01-15'] = [89800]
# x = df1_x.to_numpy()
# y = df1_y.to_numpy()
# x_pred = df2_x.to_numpy()


np.save('../data/npy/kodex1.npy',arr=([df_x, df_y]))
