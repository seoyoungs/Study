import pandas as pd
import numpy as np

df1 = pd.read_csv(r'C:\data\csv\samsung1.csv', encoding='cp949', index_col=0, header=0)
df2 = pd.read_csv(r'C:\data\csv\samsung2.csv',encoding='cp949', index_col=0, header=0)
df1 = pd.concat([df1, df2],axis = 0, join = 'outer')
df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df1.replace(',','',inplace=True, regex=True)
df1 = df1.astype('float32') #실수화
df1 = df1.sort_values(by=['일자'], axis=0) #sort_values 일자로 정렬하기

df1_x = df1.loc['2018-05-04':'2021-01-12'] #알아서 역으로 순서 바꿔준다
#df= df.iloc[::-1] 순서 알아서 바꿔줘서 안넣어도 된다.
df1_y = df1.loc['2018-05-08':'2021-01-13'] #하루 차이 나게 만들기

df1_x = df1_x.iloc[:,[0,1,2,5,6]]
df1_y = df1_y.iloc[:,[3]]
df1_y.loc['2021-01-14'] = [89700] #이걸 넣어야 y가 제대로 된 (662, 1)로 인식한다.
# print(sam2_x.head())
# print(sam2_y.head())


x = df1_x.to_numpy()
y = df1_y.to_numpy()

print(x.shape,y.shape)

# np.save('../data/npy/x_data.npy', arr=s_x)
# np.save('../data/npy/y_data.npy', arr=s_y)
np.save('../data/npy/sam_new.npy',arr=([x,y]))
# print(df1)
# merged_df = pd.concat([processed_data, data], encoding='cp949', index_col=0, header=0)

