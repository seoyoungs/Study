# 회귀

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel # feature 컬럼을 선택
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y= True) # 사이킷런에서 자동으로 x, y 부여

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

model = XGBRegressor(n_jobs=8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('r2: ', score)

thresholds = np.sort(model.feature_importances_) # 컬럼들의 값 -> sort(낮은 숫자부터 순차적으로 정렬)
print(thresholds) # 이 값들 모두 합치면 1 (컬럼 13개)
# r2:  0.9221188601856797
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

for thresh in thresholds:# 총 칼럼 13개 이므로 13번 훈련
    selection = SelectFromModel(model, threshold = thresh, prefit=True) # prefit이 True, False일때, 없을 때, 디폴트 값은

    select_x_train = selection.transform(x_train) # x_train을 select형태로 변환
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))

# feature_importance
# XGB, Tree 계열에서 많이 쓰인다

'''
(404, 13)
Thresh = 0.001, n=13, R2: 92.21%
(404, 12)
Thresh = 0.004, n=12, R2: 92.16%
(404, 11)
Thresh = 0.012, n=11, R2: 92.03%
(404, 10)
Thresh = 0.012, n=10, R2: 92.19%
(404, 9)
Thresh = 0.014, n=9, R2: 93.08%
(404, 8)
Thresh = 0.015, n=8, R2: 92.37%
(404, 7)
Thresh = 0.018, n=7, R2: 91.48%
Thresh = 0.030, n=6, R2: 92.71%
(404, 5)
Thresh = 0.042, n=5, R2: 91.74%
(404, 4)
Thresh = 0.052, n=4, R2: 92.11%
(404, 3)
(404, 2)
(404, 1)
Thresh = 0.428, n=1, R2: 44.98%
'''
