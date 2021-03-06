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
from sklearn.preprocessing import StandardScaler, RobustScaler

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
# df2 = pd.read_csv("C:/data/personal/university/2020_10_after_preprocessing.csv", low_memory = False)
# df1ch = pd.read_csv("C:/data/personal/university/2019_12_changed_compared_with_2020_10_after_preprocessing.csv", low_memory = False)
# df2ch = pd.read_csv("C:/data/personal/university/2020_10_changed_compared_with_2019_12_after_preprocessing.csv", low_memory = False)
# dfad = pd.read_csv("C:/data/personal/university/2020_10_added_compared_with_2019_12_after_preprocessing.csv", low_memory = False)
# dfdr = pd.read_csv("C:/data/personal/university/2020_10_dropped_compared_with_2019_12_after_preprocessing.csv", low_memory = False)

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
# plt.figure(figsize=(10,10))
# sns.distplot(df1['price'], rug=True,fit=norm) #distplot 히스토그램
# plt.title("Price 분포도",size=15, weight='bold')
# plt.show()

df1['price_log'] = np.log(df1.price)
# plt.figure(figsize=(12,10))
# plt.title("Log-Price 분포도",size=15, weight='bold')
# sns.distplot(df1['price_log'], rug=True,fit=norm)
# plt.show()

# Q-Q plot & boxplot
# fig = plt.figure(figsize=(14,14))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# stats.probplot(df1['price_log'], plot=plt) 
# green_diamond = dict(markerfacecolor='g', marker='D')
# ax1.boxplot(df1['price_log'], flierprops=green_diamond)
# plt.show()

#  ========= non-nominal data와 정규화 이전의 price 삭제 ==================
df1 = df1.drop(columns=['price', 'name','id' ,'host_id',
'host_name', 'description', 'neighborhood_overview',
'amenities','first_review'])

# 데이터를 요약해 보여주는pandas_profiling을 토대로 삭제
# print(df1.isnull().sum())

column_with_missing_data = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 
 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
 'review_scores_communication', 'review_scores_location', 'review_scores_value']
for i in column_with_missing_data:
    mean = df1[i].mean()
    df1[i].fillna(mean, inplace=True)
# print(df1.isnull().sum())

#cat.codes 기준
# plt.figure(figsize=(15, 8))
# mask = np.zeros_like(df1.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# palette = sns.diverging_palette(20, 220, n=256)
# corr=df1.corr(method='pearson')
# sns.heatmap(corr, annot=True, fmt=".2f", mask = mask, cmap=palette, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(22, 0)) #cat.codes 기준
# plt.title("Correlation Matrix",size=15, weight='bold')
# plt.show()

# ============== StandardScaler ================
df1_x, df1_y = df1.iloc[:,:-1], df1.iloc[:,-1] # log_price는 y값
scaler = RobustScaler()
df1_x = scaler.fit_transform(df1_x)

X_train, X_test, y_train, y_test = train_test_split(
    df1_x, df1_y, test_size=0.3,random_state=42)

lab_enc = preprocessing.LabelEncoder()
feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))

# plt.figure(figsize=(7,7))
# feat_importances = pd.Series(feature_model.feature_importances_, index=df1.iloc[:,:-1].columns)
# feat_importances.nlargest(22).plot(kind='barh') #기계 학습 모델 구축에 영향을 미치는 상위 22 가지 기능
# plt.show()

# print(lab_enc.fit_transform(y_train)) # [187 102 278 ...  39 117 207]

### Linear Regression ##
kfold_cv=KFold(n_splits=20, random_state=42, shuffle=False)
for train_index, test_index in kfold_cv.split(df1_x,df1_y):
    X_train, X_test = df1_x[train_index], df1_x[test_index]
    y_train, y_test = df1_y[train_index], df1_y[test_index]

# =================== Polynomial Transformation ================
# 과적합을 피하기 위해 각 특징의 값을 제곱하는 2차항으로 다항 변환
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)
'''
# ==================== price 예측 ================================
# =========== 1. 모든 변수 사용 ==================
##Linear Regression
# lr = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
# lr.fit(X_train, y_train)
filename = '../data/h5/hole_lr1.sav'
# pickle.dump(lr, open(filename, 'wb'))
lr = pickle.load(open(filename, 'rb'))
lr_pred= lr.predict(X_test)
print("level 1 Linear Regression")
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
#Ridge Model
# ridge_model = Ridge(alpha = 0.01, normalize = True)
# ridge_model.fit(X_train, y_train)
filename = '../data/h5/ridge_model_1.sav'
# pickle.dump(ridge_model, open(filename, 'wb'))
ridge_model = pickle.load(open(filename, 'rb'))          
pred_ridge = ridge_model.predict(X_test) 
print("level 1 Ridge Regression")
print("훈련 세트 점수: {:.2f}".format(ridge_model.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge_model.score(X_test, y_test)))
#Lasso Model
# Lasso_model = Lasso(alpha = 0.001, normalize =False)
# Lasso_model.fit(X_train, y_train)
filename = '../data/h5/Lasso_model_1.sav'
# pickle.dump(Lasso_model, open(filename, 'wb'))
Lasso_model = pickle.load(open(filename, 'rb'))  
pred_Lasso = Lasso_model.predict(X_test) 
print("level 1 Lasso Regression")
print("훈련 세트 점수: {:.2f}".format(Lasso_model.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(Lasso_model.score(X_test, y_test)))
#ElasticNet Model
# model_enet = ElasticNet(alpha = 0.01, normalize=False)
# model_enet.fit(X_train, y_train)
filename = '../data/h5/model_enet_1.sav'
# pickle.dump(model_enet, open(filename, 'wb'))
model_enet = pickle.load(open(filename, 'rb'))  
pred_test_enet= model_enet.predict(X_test)
print("level 1 ElasticNet Regression")
print("훈련 세트 점수: {:.2f}".format(model_enet.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(model_enet.score(X_test, y_test)))
print('-----------1 단계 끝 --------------')


# ============= 2. room_type 변수 제거 ========================
nyc_model_xx= df1.drop(columns=['room_type'])
nyc_model_xx, nyc_model_yx = nyc_model_xx.iloc[:,:-1], nyc_model_xx.iloc[:,-1]
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(nyc_model_xx, 
                             nyc_model_yx, test_size=0.3,random_state=42)
scaler = StandardScaler()
nyc_model_xx = scaler.fit_transform(nyc_model_xx)
kfold_cv=KFold(n_splits=20, random_state=None, shuffle=False)
for train_index, test_index in kfold_cv.split(nyc_model_xx,nyc_model_yx):
    X_train_x, X_test_x = nyc_model_xx[train_index], nyc_model_xx[test_index]
    y_train_x, y_test_x = nyc_model_yx[train_index], nyc_model_yx[test_index]
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_x = Poly.fit_transform(X_train_x)
X_test_x = Poly.fit_transform(X_test_x)

###Linear Regression
# lr_x = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
# lr_x.fit(X_train_x, y_train_x)
filename = '../data/h5/model_lr_2.sav'
# pickle.dump(lr_x, open(filename, 'wb'))
lr_x = pickle.load(open(filename, 'rb')) 
lr_pred_x= lr_x.predict(X_test_x)
print("level 2 Linear Regression")
print("훈련 세트 점수: {:.2f}".format(lr_x.score(X_train_x, y_train_x)))
print("테스트 세트 점수: {:.2f}".format(lr_x.score(X_test_x, y_test_x)))
###Ridge
# ridge_x = Ridge(alpha = 0.01, normalize = True)
# ridge_x.fit(X_train_x, y_train_x)
filename = '../data/h5/ridge_2.sav'
# pickle.dump(ridge_x, open(filename, 'wb'))
ridge_x = pickle.load(open(filename, 'rb'))        
pred_ridge_x = ridge_x.predict(X_test_x) 
print("level 2 Ridge Regression")
print("훈련 세트 점수: {:.2f}".format(ridge_x.score(X_train_x, y_train_x)))
print("테스트 세트 점수: {:.2f}".format(ridge_x.score(X_test_x, y_test_x)))
###Lasso
# Lasso_x = Lasso(alpha = 0.001, normalize =False)
# Lasso_x.fit(X_train_x, y_train_x)
filename = '../data/h5/Lasso_2.sav'
# pickle.dump(Lasso_x, open(filename, 'wb'))
Lasso_x = pickle.load(open(filename, 'rb'))
pred_Lasso_x = Lasso_x.predict(X_test_x) 
print("level 2 Lasso Regression")
print("훈련 세트 점수: {:.2f}".format(Lasso_x.score(X_train_x, y_train_x)))
print("테스트 세트 점수: {:.2f}".format(Lasso_x.score(X_test_x, y_test_x)))
##ElasticNet
# model_enet_x = ElasticNet(alpha = 0.01, normalize=False)
# model_enet_x.fit(X_train_x, y_train_x)
filename = '../data/h5/model_enet_2.sav'
# pickle.dump(model_enet_x, open(filename, 'wb'))
model_enet_x = pickle.load(open(filename, 'rb'))
pred_train_enet_x= model_enet_x.predict(X_train_x)
pred_test_enet_x= model_enet_x.predict(X_test_x)
print("level 2 ElasticNet Regression")
print("훈련 세트 점수: {:.2f}".format(model_enet_x.score(X_train_x, y_train_x)))
print("테스트 세트 점수: {:.2f}".format(model_enet_x.score(X_test_x, y_test_x)))
print('-----------2 단계 끝 --------------')


# =============== 3. neighbourhood_group_cleansed을 제거====================
nyc_model_xxx= df1.drop(columns=['neighbourhood_group_cleansed'])
nyc_model_xxx, nyc_model_yxx = nyc_model_xxx.iloc[:,:-1], nyc_model_xxx.iloc[:,-1]
X_train_xx, X_test_xx, y_train_xx, y_test_xx = train_test_split(nyc_model_xxx, 
                                  nyc_model_yxx, test_size=0.3,random_state=42)
scaler = StandardScaler()
nyc_model_xxx = scaler.fit_transform(nyc_model_xxx)
kfold_cv=KFold(n_splits=20, random_state=None, shuffle=False)
for train_index, test_index in kfold_cv.split(nyc_model_xxx,nyc_model_yxx):
    X_train_xx, X_test_xx = nyc_model_xxx[train_index], nyc_model_xxx[test_index]
    y_train_xx, y_test_xx = nyc_model_yxx[train_index], nyc_model_yxx[test_index]
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_xx = Poly.fit_transform(X_train_xx)
X_test_xx = Poly.fit_transform(X_test_xx)
###Linear Regression
# lr_xx = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
# lr_xx.fit(X_train_xx, y_train_xx)
filename = '../data/h5/lr_3.sav'
# pickle.dump(lr_xx, open(filename, 'wb'))
lr_xx = pickle.load(open(filename, 'rb'))
lr_pred_xx= lr_xx.predict(X_test_xx)
print("level 3 Linear Regression")
print("훈련 세트 점수: {:.2f}".format(lr_xx.score(X_train_xx, y_train_xx)))
print("테스트 세트 점수: {:.2f}".format(lr_xx.score(X_test_xx, y_test_xx)))
# ###Ridge
# ridge_xx = Ridge(alpha = 0.01, normalize = True)
# ridge_xx.fit(X_train_xx, y_train_xx)       
filename = '../data/h5/ridge_3.sav'
# pickle.dump(ridge_xx, open(filename, 'wb'))
ridge_xx = pickle.load(open(filename, 'rb'))    
pred_ridge_xx = ridge_xx.predict(X_test_xx) 
print("level 3 Ridge Regression")
print("훈련 세트 점수: {:.2f}".format(ridge_xx.score(X_train_xx, y_train_xx)))
print("테스트 세트 점수: {:.2f}".format(ridge_xx.score(X_test_xx, y_test_xx)))
###Lasso
# Lasso_xx = Lasso(alpha = 0.001, normalize =False)
# Lasso_xx.fit(X_train_xx, y_train_xx)
filename = '../data/h5/Lasso_3.sav'
# pickle.dump(Lasso_xx, open(filename, 'wb'))
Lasso_xx = pickle.load(open(filename, 'rb'))   
pred_Lasso_xx = Lasso_xx.predict(X_test_xx) 
print("level 3 Lasso Regression")
print("훈련 세트 점수: {:.2f}".format(Lasso_xx.score(X_train_xx, y_train_xx)))
print("테스트 세트 점수: {:.2f}".format(Lasso_xx.score(X_test_xx, y_test_xx)))
##ElasticNet
# model_enet_xx = ElasticNet(alpha = 0.01, normalize=False)
# model_enet_xx.fit(X_train_xx, y_train_xx) 
filename = '../data/h5/model_enet_3.sav'
# pickle.dump(model_enet_xx, open(filename, 'wb'))
model_enet_xx = pickle.load(open(filename, 'rb'))  
pred_train_enet_xx= model_enet_xx.predict(X_train_xx)
pred_test_enet_xx= model_enet_xx.predict(X_test_xx)
print("level 3 ElasticNet Regression")
print("훈련 세트 점수: {:.2f}".format(model_enet_xx.score(X_train_xx, y_train_xx)))
print("테스트 세트 점수: {:.2f}".format(model_enet_xx.score(X_test_xx, y_test_xx)))
print('-----------3 단계 끝 --------------')

# ============= 4. neighbourhood_group_cleansed와 room_type제거 =============
nyc_model_xxxx= df1.drop(columns=['room_type', 'neighbourhood_group_cleansed'])
nyc_model_xxxx, nyc_model_yxxx = nyc_model_xxxx.iloc[:,:-1], nyc_model_xxxx.iloc[:,-1]
X_train_xxx, X_test_xxx, y_train_xxx, y_test_xxx = train_test_split(nyc_model_xxxx, 
                               nyc_model_yxxx, test_size=0.3,random_state=42)
scaler = StandardScaler()
nyc_model_xxxx = scaler.fit_transform(nyc_model_xxxx)
kfold_cv=KFold(n_splits=20, random_state=None, shuffle=False)
for train_index, test_index in kfold_cv.split(nyc_model_xxxx,nyc_model_yxxx):
    X_train_xxx, X_test_xxx = nyc_model_xxxx[train_index], nyc_model_xxxx[test_index]
    y_train_xxx, y_test_xxx = nyc_model_yxxx[train_index], nyc_model_yxxx[test_index]
Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_xxx = Poly.fit_transform(X_train_xxx)
X_test_xxx = Poly.fit_transform(X_test_xxx)
###Linear Regression
# lr_xxx = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
# lr_xxx.fit(X_train_xxx, y_train_xxx)
filename = '../data/h5/lr_4.sav'
# pickle.dump(lr_xxx, open(filename, 'wb'))
lr_xxx = pickle.load(open(filename, 'rb'))  
lr_pred_xxx= lr_xxx.predict(X_test_xxx)
print("level 4 Linear Regression")
print("훈련 세트 점수: {:.2f}".format(lr_xxx.score(X_train_xxx, y_train_xxx)))
print("테스트 세트 점수: {:.2f}".format(lr_xxx.score(X_test_xxx, y_test_xxx)))
###Ridge
# ridge_xxx = Ridge(alpha = 0.01, normalize = True)
# ridge_xxx.fit(X_train_xxx, y_train_xxx)    
filename = '../data/h5/ridge_4.sav'
# pickle.dump(ridge_xxx, open(filename, 'wb'))
ridge_xxx = pickle.load(open(filename, 'rb'))        
pred_ridge_xxx = ridge_xxx.predict(X_test_xxx) 
print("level 4 Ridge Regression")
print("훈련 세트 점수: {:.2f}".format(ridge_xxx.score(X_train_xxx, y_train_xxx)))
print("테스트 세트 점수: {:.2f}".format(ridge_xxx.score(X_test_xxx, y_test_xxx)))
###Lasso
# Lasso_xxx = Lasso(alpha = 0.001, normalize =False)
# Lasso_xxx.fit(X_train_xxx, y_train_xxx)
filename = '../data/h5/Lasso_4.sav'
# pickle.dump(Lasso_xxx, open(filename, 'wb'))
Lasso_xxx = pickle.load(open(filename, 'rb'))
pred_Lasso_xxx = Lasso_xxx.predict(X_test_xxx) 
print("level 4 Lasso Regression")
print("훈련 세트 점수: {:.2f}".format(Lasso_xxx.score(X_train_xxx, y_train_xxx)))
print("테스트 세트 점수: {:.2f}".format(Lasso_xxx.score(X_test_xxx, y_test_xxx)))
##ElasticNet
# model_enet_xxx = ElasticNet(alpha = 0.01, normalize=False)
# model_enet_xxx.fit(X_train_xxx, y_train_xxx) 
filename = '../data/h5/model_enet_4.sav'
# pickle.dump(model_enet_xxx, open(filename, 'wb'))
model_enet_xxx = pickle.load(open(filename, 'rb'))
pred_train_enet_xxx= model_enet_xxx.predict(X_train_xxx)
pred_test_enet_xxx= model_enet_xxx.predict(X_test_xxx)
print("level 4 ElasticNet Regression")
print("훈련 세트 점수: {:.2f}".format(model_enet_xxx.score(X_train_xxx, y_train_xxx)))
print("테스트 세트 점수: {:.2f}".format(model_enet_xxx.score(X_test_xxx, y_test_xxx)))
print('-----------4 단계 끝 --------------')

# ================== 예측값 평가 ========================
print('-------------Lineer Regression-----------')
print('--Phase-1--')
print('MAE: %f'% mean_absolute_error(y_test, lr_pred))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, lr_pred)))   
print('R2 %f' % r2_score(y_test, lr_pred))
print('--Phase-2--')
print('MAE: %f'% mean_absolute_error(y_test_x, lr_pred_x))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, lr_pred_x)))   
print('R2 %f' % r2_score(y_test_x, lr_pred_x))
print('--Phase-3--')
print('MAE: %f'% mean_absolute_error(y_test_xx, lr_pred_xx))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_xx, lr_pred_xx)))   
print('R2 %f' % r2_score(y_test_xx, lr_pred_xx))
print('--Phase-4--')
print('MAE: %f'% mean_absolute_error(y_test_xxx, lr_pred_xxx))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_xxx, lr_pred_xxx)))   
print('R2 %f' % r2_score(y_test_xxx, lr_pred_xxx))
print('---------------Ridge ---------------------')
print('--Phase-1--')
print('MAE: %f'% mean_absolute_error(y_test, pred_ridge))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred_ridge)))   
print('R2 %f' % r2_score(y_test, pred_ridge))
print('--Phase-2--')
print('MAE: %f'% mean_absolute_error(y_test_x, pred_ridge_x))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_x, pred_ridge_x)))   
print('R2 %f' % r2_score(y_test_x, pred_ridge_x))
print('--Phase-3--')
print('MAE: %f'% mean_absolute_error(y_test_xx, pred_ridge_xx))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_xx, pred_ridge_xx)))   
print('R2 %f' % r2_score(y_test_xx, pred_ridge_xx))
print('--Phase-4--')
print('MAE: %f'% mean_absolute_error(y_test_xxx, pred_ridge_xxx))
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test_xxx, pred_ridge_xxx)))   
print('R2 %f' % r2_score(y_test_xxx, pred_ridge_xxx))
print('---------------Lasso-----------------------')
print('--Phase-1--')
print('MAE: %f' % mean_absolute_error(y_test, pred_Lasso))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test, pred_Lasso)))
print('R2 %f' % r2_score(y_test, pred_Lasso))
print('--Phase-2--')
print('MAE: %f' % mean_absolute_error(y_test_x, pred_Lasso_x))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x, pred_Lasso_x)))
print('R2 %f' % r2_score(y_test_x, pred_Lasso_x))
print('--Phase-3--')
print('MAE: %f' % mean_absolute_error(y_test_xx, pred_Lasso_xx))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_xx, pred_Lasso_xx)))
print('R2 %f' % r2_score(y_test_xx, pred_Lasso_xx))
print('--Phase-4--')
print('MAE: %f' % mean_absolute_error(y_test_xxx, pred_Lasso_xxx))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_xxx, pred_Lasso_xxx)))
print('R2 %f' % r2_score(y_test_xxx, pred_Lasso_xxx))
print('---------------ElasticNet-------------------')
print('--Phase-1 --')
print('MAE: %f' % mean_absolute_error(y_test,pred_test_enet))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_test_enet))) #RMSE
print('R2 %f' % r2_score(y_test, pred_test_enet))
print('--Phase-2--')
print('MAE: %f' % mean_absolute_error(y_test_x,pred_test_enet_x))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_x,pred_test_enet_x))) #RMSE
print('R2 %f' % r2_score(y_test_x, pred_test_enet_x))
print('--Phase-3--')
print('MAE: %f' % mean_absolute_error(y_test_xx,pred_test_enet_xx))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_xx,pred_test_enet_xx))) #RMSE
print('R2 %f' % r2_score(y_test_xx, pred_test_enet_xx))
print('--Phase-4--')
print('MAE: %f' % mean_absolute_error(y_test_xxx,pred_test_enet_xxx))
print('RMSE: %f' % np.sqrt(mean_squared_error(y_test_xxx,pred_test_enet_xxx))) #RMSE
print('R2 %f' % r2_score(y_test_xxx, pred_test_enet_xxx))
'''

df1_x, df1_y = df1.iloc[:,:-1], df1.iloc[:,-1] # log_price는 y값
scaler = StandardScaler()
df1_x = scaler.fit_transform(df1_x)
X_train, X_test, y_train, y_test = train_test_split(df1_x, df1_y, test_size=0.3,random_state=42)
lab_enc = preprocessing.LabelEncoder()
feature_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
feature_model.fit(X_train,lab_enc.fit_transform(y_train))
# price_model = XGBRegressor(n_estimators = 2000, learining_rate=0.01,
#                      tree_method='gpu_hist', predictor = 'gpu_predictor')
# price_model.fit(X_train, y_train, verbose=1, eval_metric=['rmse'],
#            eval_set=[(X_train, y_train), (X_test, y_test)])
# filename = '../data/h5/XGBRegressor.sav'
# # pickle.dump(model_enet_xx, open(filename, 'wb'))
# model_enet_xx = pickle.load(open(filename, 'rb')) 
# y_pred=price_model.predict(X_test)
# aaa = price_model.score(X_test, y_test)
# print('model.score : ', aaa)

parameter = [
    {'n_estimators':[400, 500, 600], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth': [4,5,6]},
    {'n_estimators':[400,600], 'learning_rate':[0.1, 0.001, 0.5],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsample_bylevel': [0.6, 0.7, 0.9]}
] #XGBRegressor : n_estimators 몇번돌릴건지

# for 문으로 묶기===========================================================
grid_random = [RandomizedSearchCV] #GridSearchCV
kflod = KFold(n_splits=5, shuffle=True)

for i in grid_random:
    model=i(XGBRegressor(n_jobs=8, tree_method='gpu_hist', 
                         predictor = 'gpu_predictor'), parameter, cv=kflod)
                        # # n_estimators == epochs

    model.fit(X_train, y_train, verbose=1, eval_metric=['rmse'],
            eval_set=[(X_train, y_train), (X_test, y_test)])
    # filename = '../data/h5/model_XGB.sav'
    # # pickle.dump(model, open(filename, 'wb'))
    # model = pickle.load(open(filename, 'rb'))
    y_pred= model.predict(X_test)
    acc=model.score(X_test, y_test)
    print('MAE: %f' % mean_absolute_error(y_test,y_pred))
    print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,y_pred))) #RMSE
    print('R2 %f' % r2_score(y_test, y_pred))
    print('model.score : ', acc)

# model.score :  0.7611791493718456

error_diff = pd.DataFrame({'Actual price': np.array(y_test).flatten(), 'Predicted price': y_pred.flatten()})
print(error_diff.head(5))









