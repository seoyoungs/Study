#######파일 앙상블해 주가 예측하기
###삼성전자 0115 파일 14열로 만들기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('C:\data\csv\삼성전자0115.csv', encoding='cp949', index_col=0, header=0, thousands=',')
df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'] # 번호부여
# df1.replace(',','',inplace=True, regex=True)
df1 = df1.drop(df1.columns[[4, 5]], axis=1) # 열 삭제axis=1(열 삭제를 뜻함)
df1 = df1.iloc[[0], :]
# print(df1)

df1 = df1.astype('float32')
df1.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df1 = df1.sort_values(by=['일자'], axis=0) # axis=0 행을 조작
 # 두개 열 삭제 했으니 번호 다시 부여
print(df1)
df1.to_csv('C:\data\csv\삼성전자0115_2.csv', index=True, encoding='cp949')
