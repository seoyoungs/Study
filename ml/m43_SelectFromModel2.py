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
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel # feature 컬럼을 선택
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y= True) # 사이킷런에서 자동으로 x, y 부여

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

model = model.best_estimator_ # 이미 모델안에 best_estimator_ 있으므로 XGB 안해도된다

for thresh in thresholds:# 총 칼럼 13개 이므로 13번 훈련
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # prefit이 True, False일때, 없을 때, 디폴트 값은
    # True인 경우 RandomizedSearchCV 사용할 수 없다
    select_x_train = selection.transform(x_train) # x_train을 select형태로 변환
    print(select_x_train.shape)

    parameter = [
        {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
        'max_depth': [4,5,6]}
    ]

    selection_model = RandomizedSearchCV(model, parameter)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))

# feature_importance
# XGB, Tree 계열에서 많이 쓰인다
print(model.coef_) # 기울기 절편 XGB에서는 안된다
print(model.intercept) # XGB에서 사용이 될까?
#  AttributeError: Coefficients are not defined for Booster type None(부스터에서는 지원 안된다)

'''
밑에만 수정
selection_model = GridSearchCV(XGBRegressor(n_jobs=8), parameter)
(404, 13)
Thresh = 0.001, n=13, R2: 91.88%
(404, 12)
Thresh = 0.004, n=12, R2: 93.28%
(404, 11)
Thresh = 0.012, n=11, R2: 92.86%
(404, 10)
Thresh = 0.012, n=10, R2: 92.94%
(404, 9)
Thresh = 0.014, n=9, R2: 93.35%
(404, 8)
Thresh = 0.015, n=8, R2: 93.26%
(404, 7)
Thresh = 0.018, n=7, R2: 93.48%
(404, 6)
Thresh = 0.030, n=6, R2: 92.69%
(404, 5)
Thresh = 0.042, n=5, R2: 92.52%
(404, 4)
Thresh = 0.052, n=4, R2: 92.41%
(404, 3)
Thresh = 0.069, n=3, R2: 91.81%
(404, 2)
Thresh = 0.301, n=2, R2: 82.52%
(404, 1)
Thresh = 0.428, n=1, R2: 70.59%
위아래 모두 수정
Thresh = 0.002, n=13, R2: 93.28%
(404, 12)
Thresh = 0.006, n=12, R2: 91.97%
(404, 11)
Thresh = 0.010, n=11, R2: 92.86%
(404, 10)
Thresh = 0.012, n=10, R2: 92.79%
(404, 9)
Thresh = 0.016, n=9, R2: 93.35%
(404, 8)
Thresh = 0.021, n=8, R2: 93.20%
(404, 7)
Thresh = 0.027, n=7, R2: 92.53%
(404, 6)
Thresh = 0.033, n=6, R2: 92.14%
(404, 5)
Thresh = 0.036, n=5, R2: 92.69%
(404, 4)
Thresh = 0.045, n=4, R2: 91.43%
(404, 3)
Thresh = 0.054, n=3, R2: 88.32%
(404, 2)
Thresh = 0.273, n=2, R2: 82.52%
(404, 1)
Thresh = 0.465, n=1, R2: 70.59%
'''
'''
# -----------------------------------------------------최적의 파라미터 --------
# # 3. 위 피처 갯수로 데이터(피처)을 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용해
# 최적의 R2구할 것
# x_train을 선택한 Feature로 줄입니다.
selection = SelectFromModel(model, 
                            threshold = 0.016, #Feature 선택에 사용할 임계 값, 중요도가 크거나 같은 기능은 유지되고 나머지는 삭제
                            prefit=True        #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                            )
selection_x_train = selection.transform(x_train)
print(selection_x_train.shape)
selection_model = GridSearchCV(model, parameter)
selection_model.fit(selection_x_train, y_train)
selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)
score = r2_score(y_test, y_predict)
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.016, selection_x_train.shape[1], score*100))
# Thresh=0.025, n=6, R2:90.96%
selection_model = selection_model.best_estimator_
selection_model.fit(selection_x_train, y_train)
selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)
score = r2_score(y_test, y_predict)
print('Thresh=%.3f, n=%d, R2:%.2f%%' %(0.016, selection_x_train.shape[1], score*100))
# Thresh=0.016, n=9, R2:93.35%
# Thresh=0.016, n=9, R2:93.35%
'''

