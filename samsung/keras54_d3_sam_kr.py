import numpy as np
import pandas as pd

df = pd.read_csv('C:\data\csv\samsung.csv', index_col=0,header=0,encoding='CP949')

# 10, 13

df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')
df = df.sort_values(by=['일자'], axis=0)


# # 상관 계수 시각화!
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"


sns.set(font_scale=1.2, font='Malgun Gothic')
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()
