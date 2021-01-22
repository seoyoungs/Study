import numpy as np
import pandas as pd
import re


df = pd.read_csv('../data/csv/samsung.csv',index_col=0,header=0, encoding='cp949')
df_2 = pd.read_csv('../data/csv/sam1104.csv',index_col=0,header=0, encoding='cp949')
df_3 = pd.read_csv(r'C:\data\csv\삼성전자0115.csv',encoding='cp949', index_col=0, header=0)

df.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
df_2.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']

df.replace(',','',inplace=True, regex=True)
df_2.replace(',','',inplace=True, regex=True)
df_3.replace(',','',inplace=True, regex=True)
df = df.astype('float32')
df_2 = df_2.astype('float32')
df_3 = df_3.astype('float32')
df = df.sort_values(by=['일자'],axis=0)
df_2 = df_2.sort_values(by=['일자'],axis=0)
df_3 = df_3.sort_values(by=['일자'],axis=0)
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
# sns.set(font_scale = 1.2)
# sns.heatmap(data=df.corr(), square =True, annot=True, cbar = True)
# # plt.rcParams['axes.unicode_minus'] = False 
# sns.set(font = 'Malgun Gothic', rc= {'axes.unicode_minus':False},style='darkgrid')
# plt.show()

# print(df)

df_x = df.loc['2018-05-04': '2021-01-13']
df_y = df.loc['2018-05-08' : '2021-01-13']
# df_x1 = df.loc['2021-01-13': ]
df_2_x = df_2.loc['2021-01-14':'2021-01-14' ]
# print(df_x.shape)   # 661, 14
# print(df_y.shape)   # 661, 14

df_x = df_x.iloc[:,[0,1,2,9,12]]
# df_x1 = df_x1.iloc[:,[0,1,2,9,12]]
df_2_x = df_2_x.iloc[:,[0,1,2,11,14]]
# print(df_2_x)

df_y = df_y.iloc[:,[3]]
df_y.loc['2021-01-14'] = [89700]

# print(df_y.shape)   #662,1
# print(df_x.shape)   #662,5

x= df_x.to_numpy()
y = df_y.to_numpy()
x_pred = df_2_x.to_numpy()

# print(df_x[-1:])
'''
2021-01-13  89800.0  91200.0  89100.0 -1781416.0 -2190214.0
'''
# print(df_y[-2:])


# x = dataset.data
# y = dataset.target

np.save('../data/npy/sam2.npy',arr=([x,y,x_pred]))