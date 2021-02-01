# 데이터 별로 5개 만든다
# cross_val 사용 pipe는 안해도 됨

from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_diabetes, load_iris,load_wine, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier,XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kflod = KFold(n_splits=5, shuffle=True)


parameter = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsample_bylevel': [0.6, 0.7, 0.9]}
]

model = GridSearchCV(XGBRegressor(eval_metric='mlogloss'), parameter, cv=kflod)
score = cross_val_score(model, x_train, y_train, cv= kflod) #GridSearchCV에서 나온값, 또 5번해서 최적의 값
# 그럼 총 25번
print(score.shape)
print('교차검증점수 : ', score)

# 훈련
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)


