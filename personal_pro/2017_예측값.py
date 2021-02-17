# https://www.kaggle.com/kimbaekseyeong/preprocessing-airbnb-data-2019-12-and-2020-10
# https://www.kaggle.com/inahsong/nyc-airbnb-price-prediction-2019-12-and-2020-10/data?select=2020_10_dropped_compared_with_2019_12_after_preprocessing.csv

import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier, XGBRegressor

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor

# pandas to_csv() function
df1 = pd.read_csv("C:/data/personal/university/2019_12_after_preprocessing.csv", low_memory = False)
# print(df1.describe())

# ======= 세부지역 & 룸타입 numeric으로 변환 ==============
df1['neighbourhood_group_cleansed']= df1['neighbourhood_group_cleansed'].astype("category").cat.codes
df1['neighbourhood_cleansed'] = df1['neighbourhood_cleansed'].astype("category").cat.codes
df1['room_type'] = df1['room_type'].astype("category").cat.codes
# print(df1.info())

# ============ 종속변수인 price 칼럼을 정규분포 ====================
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"

df1['price_log'] = np.log(df1.price)

#  ========= non-nominal data와 정규화 이전의 price 삭제 ==================
df1 = df1.drop(columns=['price','name', 'id' ,'host_id','host_name', 'description', 'neighborhood_overview','amenities','first_review']) #'price', name
# 데이터를 요약해 보여주는pandas_profiling을 토대로 삭제
# print(df1.isnull().sum())

column_with_missing_data = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
for i in column_with_missing_data:
    mean = df1[i].mean()
    df1[i].fillna(mean, inplace=True)
# print(df1.isnull().sum())

'''
# ============== StandardScaler ================
df1_x, df1_y = df1.iloc[:,:-1], df1.iloc[:,-1] # log_price는 y값
scaler = StandardScaler()
df1_x = scaler.fit_transform(df1_x)

X_train, X_test, y_train, y_test = train_test_split(df1_x, df1_y, test_size=0.3,random_state=42)

lab_enc = preprocessing.LabelEncoder()

feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))


kfold_cv=KFold(n_splits=20, random_state=42, shuffle=False)
for train_index, test_index in kfold_cv.split(df1_x,df1_y):
    X_train, X_test = df1_x[train_index], df1_x[test_index]
    y_train, y_test = df1_y[train_index], df1_y[test_index]

# 과적합을 피하기 위해 각 특징의 값을 제곱하는 2차항으로 다항 변환
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)

# 원래 값과 그거에 대한 예측값 비교
# https://www.kaggle.com/nageshsingh/airbnb-price-prediction
#Gradient Boosting Regressor
#Prepare a Linear Regression Model

df1_1 = df1[['accommodates', 'price_log']].groupby(['accommodates'], as_index=False).mean().sort_values(by='price_log',ascending=False)
print(df1_1.head(n=10))
df1_2 = df1[['room_type', 'price_log']].groupby(['room_type'], as_index=False).mean().sort_values(by='price_log',ascending=False)
print(df1_2.head(n=10))

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
#R2 score
from sklearn.metrics import r2_score, mean_squared_error
print("R2 score: ",r2_score(y_test,y_pred)*100,"%")
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

#Error
error_diff = pd.DataFrame({'Actual price': np.array(y_test).flatten(), 'Predicted price': y_pred.flatten()})
print(error_diff.head(5))

#Visualize the error
# df1 = error_diff.head(25)
# df1.plot(kind='bar',figsize=(10,7))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()
'''


# ============================= 방이 1개인 경우 값 예측 ===============================
reqd_Index = df1[df1['room_type'] == 1]
print(reqd_Index)

df1_x, df1_y = reqd_Index.iloc[:,:-1], reqd_Index.iloc[:,-1] # log_price는 y값
scaler = StandardScaler()
reqd_Index = scaler.fit_transform(df1_x)

X_train, X_test, y_train, y_test = train_test_split(df1_x, df1_y, test_size=0.3,random_state=42)

lab_enc = preprocessing.LabelEncoder()

feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))


# kfold_cv=KFold(n_splits=20, random_state=42, shuffle=False)
# for train_index, test_index in kfold_cv.split(df1_x,df1_y):
#     X_train, X_test = df1_x[train_index], df1_x[test_index]
#     y_train, y_test = df1_y[train_index], df1_y[test_index]

# 과적합을 피하기 위해 각 특징의 값을 제곱하는 2차항으로 다항 변환
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)

# 원래 값과 그거에 대한 예측값 비교
# https://www.kaggle.com/nageshsingh/airbnb-price-prediction
#Gradient Boosting Regressor
#Prepare a Linear Regression Model

# df1_1 = reqd_Index[['accommodates', 'price_log']].groupby(['accommodates'], as_index=False).mean().sort_values(by='price_log',ascending=False)
# print(df1_1.head(n=10))
# df1_2 = reqd_Index[['room_type', 'price_log']].groupby(['room_type'], as_index=False).mean().sort_values(by='price_log',ascending=False)
# print(df1_2.head(n=10))

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
#R2 score
# from sklearn.metrics import r2_score, mean_squared_error
# print("R2 score: ",r2_score(y_test,y_pred)*100,"%")
# print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# #Error
# error_diff = pd.DataFrame({'Actual price': np.array(y_test).flatten(), 'Predicted price': y_pred.flatten()})
# print(error_diff.head(5))



feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))

price_model = XGBRegressor(n_estimators = 1000, learining_rate=0.01,
                     tree_method='gpu_hist', predictor = 'gpu_predictor')
price_model.fit(X_train, y_train, verbose=1, eval_metric=['rmse'],
           eval_set=[(X_train, y_train), (X_test, y_test)])
y_pred=reg.predict(X_test)
#R2 score
aaa = price_model.score(X_test, y_test)
print('model.score : ', aaa)

#Error
error_diff = pd.DataFrame({'Actual price': np.array(y_test).flatten(), 'Predicted price': y_pred.flatten()})
print(error_diff.head(5))



