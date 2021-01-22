import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('C:\data\csv\samsung.csv', encoding='cp949', index_col=0, header=0)
#print(sam.head())

df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')
df = df.sort_values(by=['일자'], axis=0)
df= df.iloc[::-1]
df = df.drop("2021-01-13") # df = df.drop("c1", axis=0)

# print(df.head())
df.to_csv('C:\data\csv\samsung1.csv', index=True, encoding='cp949')

'''
df_x = df.loc['2018-05-04':'2021-01-12']
df_y = df.loc['2018-05-08':'2021-01-13']

df_x = df_x.iloc[:,[0,1,2,5,6]]
# (661,7)

df_y = df_y.iloc[:,[3]]

x_data = df_x.to_numpy()
y_data = df_y.to_numpy()

np.save('../data/npy/x1_data.npy', arr=x_data)
np.save('../data/npy/y2_data.npy', arr=y_data)
'''

