### 회귀랑 분류랑 다른 곳 eval_metric = (rmse, error), (XGBRegressor, XGBClassifier), (XGBRegressor, XGBClassifier)
### load_breast_cancer (분류)

## 이발셋
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# x, y = load_boston(return_X_y=True) #예제 데이터 땡겨올때 쓴다
dataset = load_breast_cancer()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, shuffle=True, random_state = 66
)
# print(y) #0,1로 이진분류

# 2. 모델
model = XGBClassifier(n_estimators = 50, learning_rate = 0.01, n_jobs=8)    # 이렇게 튜닝 할 줄 알기

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss', 'error'],
          eval_set=[(x_train, y_train), (x_test, y_test)])       # eval_metric='logloss' 이진분류인 경우
# eval_metric = logloss, acc, mlogloss 다 분류에 쓰인다

# 평가
aaa = model.score(x_test, y_test)
print('aaa: ', aaa)

y_pred = model.predict(x_test)
r2 = accuracy_score(y_test, y_pred)        #score 할 때 원데이터 (y_test를 앞에 넣기)
print('accuracy_score : ', accuracy_score)

## 이발셋 =============================
print('====================================')
result = model.evals_result()
print('result: ', result)