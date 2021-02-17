# https://www.kaggle.com/kimbaekseyeong/preprocessing-airbnb-data-2019-12-and-2020-10
# https://www.kaggle.com/inahsong/nyc-airbnb-price-prediction-2019-12-and-2020-10/data?select=2020_10_dropped_compared_with_2019_12_after_preprocessing.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

df1 = pd.read_csv("C:/data/personal/university/2019_12_listing_after_extraction.csv", low_memory = False)
df2 = pd.read_csv("C:/data/personal/university/2020_10_listing_after_extraction.csv", low_memory = False)
df1ch = pd.read_csv("C:/data/personal/university/2019_12_changed_compared_with_2020_10.csv", low_memory = False)
df2ch = pd.read_csv("C:/data/personal/university/2020_10_changed_compared_with_2019_12.csv", low_memory = False)
dfad = pd.read_csv("C:/data/personal/university/2020_10_added_compared_with_2019_12.csv", low_memory = False)
dfdr = pd.read_csv("C:/data/personal/university/2019_12_dropped_compared_with_2020_10.csv", low_memory = False)

print(df1['price'].head(5))
# replace() function
df1['price'] = df1['price'].str.replace(',', '')
df1['price'] = df1['price'].str.replace('$', '')
df1['price'] = df1['price'].astype(float)
df2['price'] = df2['price'].str.replace(',', '')
df2['price'] = df2['price'].str.replace('$', '')
df2['price'] = df2['price'].astype(float)
df1ch['price'] = df1ch['price'].str.replace(',', '')
df1ch['price'] = df1ch['price'].str.replace('$', '')
df1ch['price'] = df1ch['price'].astype(float)
df2ch['price'] = df2ch['price'].str.replace(',', '')
df2ch['price'] = df2ch['price'].str.replace('$', '')
df2ch['price'] = df2ch['price'].astype(float)
dfad['price'] = dfad['price'].str.replace(',', '')
dfad['price'] = dfad['price'].str.replace('$', '')
dfad['price'] = dfad['price'].astype(float)
dfdr['price'] = dfdr['price'].str.replace(',', '')
dfdr['price'] = dfdr['price'].str.replace('$', '')
dfdr['price'] = dfdr['price'].astype(float)

# info() function
# df1.info()
# df2.info()
# df1ch.info()
# df2ch.info()
# dfad.info()
# dfdr.info()

# =============== 결측 값 제거 =========================
df1 = df1.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type',
  'accommodates','amenities','price','minimum_nights','availability_365',
   'number_of_reviews','first_review','reviews_per_month'])
df2 = df2.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type','accommodates',
  'amenities','price','minimum_nights','availability_365','number_of_reviews','first_review','reviews_per_month'])
df1ch = df1ch.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type','accommodates',
  'amenities','price','minimum_nights','availability_365','number_of_reviews',
  'first_review','reviews_per_month'])
df2ch = df2ch.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type','accommodates',
  'amenities','price','minimum_nights','availability_365','number_of_reviews',
  'first_review','reviews_per_month'])
dfad = dfad.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type','accommodates',
  'amenities','price','minimum_nights','availability_365','number_of_reviews',
  'first_review','reviews_per_month'])
dfdr = dfdr.dropna(subset=['name','host_id','host_name','neighbourhood_cleansed',
  'neighbourhood_group_cleansed','latitude','longitude','room_type','accommodates',
  'amenities','price','minimum_nights','availability_365','number_of_reviews',
  'first_review','reviews_per_month'])

# 결측 값 제거 확인
# print(df1.isnull().sum())
#df2.isnull().sum()
#df1ch.isnull().sum()
#df2ch.isnull().sum()
#dfad.isnull().sum()
#dfdr.isnull().sum()

# price와 availability_365의 값이 0인 low를 모든 데이터 프레임에서 제거
df1 = df1[df1.price != 0]
df1 = df1[df1.availability_365 != 0]
df2 = df2[df2.price != 0]
df2 = df2[df2.availability_365 != 0]
df1ch = df1ch[df1ch.price != 0]
df1ch = df1ch[df1ch.availability_365 != 0]
df2ch = df2ch[df2ch.price != 0]
df2ch = df2ch[df2ch.availability_365 != 0]
dfad = dfad[dfad.availability_365 != 0]
dfdr = dfdr[dfdr.price != 0]
dfdr = dfdr[dfdr.availability_365 != 0]

# price, minimum_nights가 지나치게 높을 때 Outlier 제거
# =========== price 가격 제거 ============================
p = [i for i in range(100)]
content = np.percentile(df1['price'], p, interpolation='nearest') # percentile 퍼센트
# print(len(content)) # 100
# print percent of value
def price_percentage_to_string(start, stop, step):
    p = np.arange(start, stop, step)
    prev = 'start'
    try:
        for i in p:
            if prev != content[i]:
                msg = '가격이 ' + str(content[i]) + '$ 이내에 분포하는 숙소는 전체 데이터의 ' + str(i) + '퍼센트입니다.'
                print(msg)
                prev = content[i]
    except IndexError:
        pass
# print(price_percentage_to_string(0, 100, 1))

#가격이 525$ 이상인 데이터의 수 확인 및 출력
temp = sum(df1.price > 525)
print(temp) # 499

#가격이 525$ 이상인 데이터 삭제 및 출력
df1 = df1[df1.price <= 525]
print(sum(df1.price <= 525)) # 24819

# ============ 숙박일수 제거 ======================
#최소 숙박 일수가 45일 이상인 데이터의 수 확인 및 출력
temp = sum(df1.minimum_nights > 90)
print(temp) # 144

#최소 숙박 일수가 45일 이상인 데이터 삭제 및 출력
df1 = df1[df1.minimum_nights <= 90]
print(sum(df1.minimum_nights <= 90)) # 24675
# df1는 총 543개의 low가 삭제되었으며, 24675개의 low가 남았습니다.







