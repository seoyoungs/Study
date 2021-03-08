# tree_method='gpu_hist' 를 넣으면 된다
# predictor = 'gpu_predictor' --> 훈련도 gpu쓴다

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 66
)

model = XGBRegressor(n_estimators = 100000, learining_rate=0.01,
                     tree_method='gpu_hist', predictor = 'gpu_predictor')

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
           eval_set=[(x_train, y_train), (x_test, y_test)],
           early_stopping_rounds=3000)

# predictor  = 'gpu_predictor' 인 부분
aaa = model.score(x_test, y_test)
print('model.score : ', aaa)