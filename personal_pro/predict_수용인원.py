# https://www.kaggle.com/kimbaekseyeong/preprocessing-airbnb-data-2019-12-and-2020-10
# https://www.kaggle.com/inahsong/nyc-airbnb-price-prediction-2019-12-and-2020-10/data?select=2020_10_dropped_compared_with_2019_12_after_preprocessing.csv

import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
# df1 = pd.read_csv("C:/data/personal/university/2019_12_after_preprocessing.csv", low_memory = False)
# print(df1.describe())
# ============================= 수용인원 2명인 경우 값 예측 ===============================
# reqd_Index_1 = df1[df1['accommodates'] == 2]
# reqd_Index_1.to_csv("C:/data/personal/university/2019_12_accommodates.csv", index = False)
reqd_Index = pd.read_csv("C:/data/personal/university/2019_12_accommodates.csv", low_memory = False)

# ======= 세부지역 & 룸타입 numeric으로 변환 ==============
reqd_Index['neighbourhood_group_cleansed']= reqd_Index['neighbourhood_group_cleansed'].astype("category").cat.codes
reqd_Index['neighbourhood_cleansed'] = reqd_Index['neighbourhood_cleansed'].astype("category").cat.codes
reqd_Index['room_type'] = reqd_Index['room_type'].astype("category").cat.codes
# print(df1.info())

reqd_Index['price_log'] = np.log(reqd_Index.price)

#  ========= non-nominal data와 정규화 이전의 price 삭제 ==================
reqd_Index = reqd_Index.drop(columns=['price','name', 'id' ,'host_id','host_name', 'description', 'neighborhood_overview','amenities','first_review']) #'price', name
# 데이터를 요약해 보여주는pandas_profiling을 토대로 삭제
# print(df1.isnull().sum())

column_with_missing_data = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
for i in column_with_missing_data:
    mean = reqd_Index[i].mean()
    reqd_Index[i].fillna(mean, inplace=True)
# print(df1.isnull().sum())

# print(reqd_Index)

# ============== StandardScaler ================

df1_x, df1_y = reqd_Index.iloc[:,:-1], reqd_Index.iloc[:,-1] # log_price는 y값
scaler = StandardScaler()
df1_x = scaler.fit_transform(df1_x)

X_train, X_test, y_train, y_test = train_test_split(
    df1_x, df1_y, test_size=0.3,random_state=42)

lab_enc = preprocessing.LabelEncoder()
feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))

# for 문으로 묶기===========================================================
parameter = [
    {'n_estimators':[400, 500, 600], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth': [4,5,6]},
    {'n_estimators':[400,600], 'learning_rate':[0.1, 0.001, 0.5],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsample_bylevel': [0.6, 0.7, 0.9]}
] #XGBRegressor : n_estimators 몇번돌릴건지

grid_random = [RandomizedSearchCV] #GridSearchCV
kflod = KFold(n_splits=5, shuffle=True)

for i in grid_random:
    model=i(XGBRegressor(n_jobs=8, tree_method='gpu_hist', 
                         predictor = 'gpu_predictor'), parameter, cv=kflod)
                         #n_estimators == epochs

    model.fit(X_train, y_train, verbose=1, eval_metric=['rmse'],
            eval_set=[(X_train, y_train), (X_test, y_test)])
    filename = '../data/h5/model_XGB_person.sav'
    pickle.dump(model, open(filename, 'wb'))
    # model = pickle.load(open(filename, 'rb'))
    y_pred= model.predict(X_test)
    acc=model.score(X_test, y_test)
    print('MAE: %f' % mean_absolute_error(y_test,y_pred))
    print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
    print('R2 %f' % r2_score(y_test, y_pred))
    print('model.score : ', acc)


'''
price_model = XGBRegressor(n_estimators = 580, learining_rate=0.01,
                     tree_method='gpu_hist', predictor = 'gpu_predictor')
price_model.fit(X_train, y_train, verbose=1, eval_metric=['rmse'],
           eval_set=[(X_train, y_train), (X_test, y_test)])
y_pred=price_model.predict(X_test)

#R2 score
aaa = price_model.score(X_test, y_test)
print('model.score : ', aaa)
'''
#Error
error_diff = pd.DataFrame({'Actual price': np.array(y_test).flatten(), 'Predicted price': y_pred.flatten()})
print(error_diff.head(5))

'''
MAE: 0.230905
RMSE: 0.305842
R2 0.652184
model.score :  0.6521841708828349
   Actual price  Predicted price
0      4.779123         4.983882
1      4.976734         5.120294
2      4.941642         4.975259
3      4.127134         4.128029
4      4.369448         4.586910
'''



