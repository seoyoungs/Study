import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sam = pd.read_csv('C:\data\csv\samsung.csv', encoding='cp949', index_col=0, header=0)
#print(sam.head())

sam1= sam.loc[::-1] # 데이터 일자 역순으로 뒤집기
#print(sam1.head())

#sam2=sam1[1738:]  #날짜 자르기
#print(sam2.head())
#print(sam2.tail())
sam2=sam1.iloc[:, 0:7] # 열자르기
#sam2 = sam2.drop(columns='등락률') # 열 등락률 삭제
#print(sam3.head())

sam2.columns = ['1', '2', '3', '4','5', '6', '7']
sam2.replace(',','',inplace=True, regex=True)
sam2 = sam2.astype('float32')
sam2 = sam2.sort_values(by=['일자'], axis=0)
#print(sam2.head())

sam2_x = sam2.loc['2018-05-04':'2021-01-12']
sam2_y = sam2.loc['2018-05-08':'2021-01-13'] #하루 차이 나게 만들기

sam2_x = sam2_x.iloc[:,[0,1,2,5,6]]
sam2_y = sam2_y.iloc[:,[3]]
# print(sam2_x.head())
# print(sam2_y.head())

s_x = sam2_x.to_numpy()
s_y = sam2_y.to_numpy()

np.save('../data/npy/x_data.npy', arr=s_x)
np.save('../data/npy/y_data.npy', arr=s_y)