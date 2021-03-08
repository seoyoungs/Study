## 이발셋
### 회귀랑 분류랑 다른 곳 eval_metric = (rmse, error), (XGBRegressor, XGBClassifier)
## load_boston 회귀

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# x, y = load_boston(return_X_y=True) #예제 데이터 땡겨올때 쓴다
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, shuffle=True, random_state = 66
)

# 2. 모델
model = XGBRegressor(n_estimators = 100, learning_rate = 0.01, n_jobs=8)    # 이렇게 튜닝 할 줄 알기

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',
          eval_set=[(x_train, y_train), (x_test, y_test)])       # eval_metric='rmse' 회귀인 경우

# 평가
aaa = model.score(x_test, y_test)
print('aaa: ', aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)        #score 할 때 원데이터 (y_test를 앞에 넣기)
print('r2 : ', r2)

## 이발셋 =============================
print('====================================')
result = model.evals_result()
print('result: ', result)

'''
n_estimators = 10 일 때
result:  {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225,
22.947048, 22.730053, 22.515182, 22.302441, 22.091829, 21.883278, 21.676794])]), 
'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978, 23.100504, 
22.87919, 22.660995, 22.444965, 22.23027, 22.018494, 21.808922])])}
'''
