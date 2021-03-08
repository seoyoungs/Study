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
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 20)       # eval_metric='rmse' 회귀인 경우

# 평가
aaa = model.score(x_test, y_test)
print('score: ', aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)        #score 할 때 원데이터 (y_test를 앞에 넣기)
print('r2 : ', r2)

## 이발셋 =============================
print('====================================')
result = model.evals_result()
# print('result: ', result)

#저장  ======================================
import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb')) # 저장 'wb'
# print('저장완료') # 모델 save

# 불러오기
model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb')) # 불러오기 'rb'
print("불러왔다")
r22 = model2.score(x_test, y_test)
print('r22 : ', r22)

