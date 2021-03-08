## 이발셋
### 회귀랑 분류랑 다른 곳 eval_metric = (rmse, error), (XGBRegressor, XGBClassifier), (r2_score, accuracy_score)
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
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, shuffle=True, random_state = 66
)

# 2. 모델
model = XGBRegressor(n_estimators = 1000, learning_rate = 0.01, n_jobs=8)    # 이렇게 튜닝 할 줄 알기

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss','rmse'],
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 50)       # eval_metric='rmse' 회귀인 경우

# 평가
aaa = model.score(x_test, y_test)
print('score: ', aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)        #score 할 때 원데이터 (y_test를 앞에 넣기)
print('r2 : ', r2)

## 이발셋 =============================
print('====================================')
result = model.evals_result()
print('result: ', result)

import matplotlib.pyplot as plt
# eval_metric=['rmse', 'logloss'] 순서대로

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)


fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('Rmse') # 맨마지막으로 해야 그래프가 선형이 된다
plt.title('XGBoost RMSE')
plt.show()
