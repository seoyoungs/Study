import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('C:\data\csv\삼성전자0114.csv', encoding='cp949', index_col=0, header=0)
df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df1.replace(',','',inplace=True, regex=True)
df1 = df1.astype('float32')
df1 = df1.sort_values(by=['일자'], axis=0)
df1= df1.iloc[::-1]
# print(df1.head())

df1.to_csv('C:\data\csv\samsung2.csv', index=True, encoding='cp949')

