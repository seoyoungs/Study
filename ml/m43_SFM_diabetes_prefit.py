######### ml43_2 카피 

# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2와 피처임포턴스 구할 것

# 2. 위 쓰레드 값으로 SelectiFromModel을 구해서
# 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)을 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용해
# 최적의 R2구할 것

# 1번값과 2번값 비교

# 회귀

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel # feature 컬럼을 선택
from sklearn.metrics import r2_score, accuracy_score

x, y = load_diabetes(return_X_y= True) # 사이킷런에서 자동으로 x, y 부여

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

parameter = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth': [4,5,6]}
]

model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameter)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('r2: ', score)

thresholds = np.sort(model.best_estimator_.feature_importances_) # 컬럼들의 값 -> sort(낮은 숫자부터 순차적으로 정렬)
# print(model.best_estimator_.feature_importances_)
print(thresholds) # 이 값들 모두 합치면 1 (컬럼 13개)
# r2:  0.9188116974777065
#  0.02678531 0.03278282 0.03606399 0.04534625 0.05393368 0.27339098
#  0.4654915 ]
'''
# ================= prefit=True 일 때 ========================================
for thresh in thresholds:# 총 칼럼 13개 이므로 13번 훈련
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit=True) # prefit이 True, False일때, 없을 때, 디폴트 값은
    # True인 경우 RandomizedSearchCV 사용할 수 없다
    select_x_train = selection.transform(x_train) # x_train을 select형태로 변환
    print(select_x_train.shape)

    parameter = [
        {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
        'max_depth': [4,5,6]}
    ]

    selection_model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameter)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
#======================================================================================
'''

# ============== prefit = False일 때 defalut -> .fit해주기 ==============================

for thresh in thresholds:# 총 칼럼 13개 이므로 13번 훈련
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit=False).fit(x_train, y_train) # prefit이 True, False일때, 없을 때, 디폴트 값은
    # True인 경우 RandomizedSearchCV 사용할 수 없다
    select_x_train = selection.transform(x_train) # x_train을 select형태로 변환
    print(select_x_train.shape)

    parameter = [
        {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
        'max_depth': [4,5,6]}
    ]

    selection_model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameter)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
# ==============================================================
# feature_importance
# XGB, Tree 계열에서 많이 쓰인다

'''
acc증가 시키기
(353, 10)
Thresh = 0.038, n=10, R2: 27.14%
(353, 9)
Thresh = 0.042, n=9, R2: 32.59%
(353, 8)
Thresh = 0.044, n=8, R2: 34.23%
(353, 7)
Thresh = 0.045, n=7, R2: 23.01%
(353, 6)
Thresh = 0.055, n=6, R2: 28.72%
(353, 5)
Thresh = 0.065, n=5, R2: 32.55%
(353, 4)
Thresh = 0.076, n=4, R2: 28.87%
(353, 3)
Thresh = 0.090, n=3, R2: 32.90%
(353, 2)
Thresh = 0.185, n=2, R2: 36.29%
(353, 1)
Thresh = 0.359, n=1, R2: 22.15%
'''
