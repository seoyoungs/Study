import pandas as pd
import numpy as np

df1 = pd.read_csv(r'C:\data\csv\samsung1.csv', encoding='cp949', index_col=0, header=0)
df2 = pd.read_csv(r'C:\data\csv\samsung2.csv',encoding='cp949', index_col=0, header=0) #데이터2개
df3 = pd.read_csv(r'C:\data\csv\삼성전자0115_2.csv',encoding='cp949', index_col=0, header=0)

df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df2.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df3.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

df = pd.concat([df3,df2,df1], join='inner')
df = df.sort_index(ascending=True)
# print(df)


# df1.replace(',','',inplace=True, regex=True)
df = df.astype('float64') #실수화
df = df.sort_values(by=['일자'], axis=0)
df=df[1738:]
df_x = df.iloc[:-1,[1,2,3,5,6,8]]
df_y = df.iloc[1:,[0]]

df_y.loc['2021-01-15'] = [89800]

print(df_x.shape)
print(df_y.shape)
# df = df.sort_index(ascending=False) 
# print(df_x) #sort_values 일자로 정렬하기
df_x = df_x.to_numpy()
df_y = df_y.to_numpy()

# data1 = df.to_numpy

# #df= df.iloc[::-1] 순서 알아서 바꿔줘서 안넣어도 된다.
# df_y = df.loc['2018-05-08':'2021-01-15'] #하루 차이 나게 만들기
# df2_x = df.loc['2021-01-15':'2021-01-15' ]

# print(df1_x)
# df1_x = df1_x.iloc[:,[1,2,3,5,6,12]]
# df2_x = df2_x.iloc[:,[1,2,3,5,6,12]]
# df_y = df_y.iloc[:,[0]]
# # print(df1_y.shape)
# df_y.loc['2021-01-15'] = [89800] #이걸 넣어야 y가 제대로 된 (662, 1)로 인식한다.
# print(df1_x)
# print(df1_x.shape)

# x1 = df1_x.to_numpy()
# y1 = df_y.to_numpy()
# x_pred = df2_x.to_numpy()

np.save('../data/npy/sam_new2.npy',arr=([df_x, df_y]))
# df.to_csv('../data/csv/samsung_123.csv', index=True, encoding='cp949')




