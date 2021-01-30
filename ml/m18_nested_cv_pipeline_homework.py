## 데이터는 randomForest 쓰고
# 파이프라인 엮어서 25번 돌리기
# 데이터는 diabets
# http://blog.naver.com/PostView.nhn?blogId=winddori2002&logNo=221659080425&parentCategoryNo=1&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #격자형으로하고 크로스형으로도 한다는 뜻
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

scaler = StandardScaler()
rf = RandomForestRegressor()

param_grid = {
    "clf__n_estimators": [100, 500, 1000],
    # "clf__max_depth": [1, 5, 10, 25],
    # "clf__max_features": [*np.arange(0.1, 1.1, 0.1)],
}

for parameters in param_grid:
    pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestRegressor())
    ])

    model = GridSearchCV(pipe, param_grid, cv=kflod, n_jobs=-1, verbose=1)
    score = cross_val_score(model, x_train, y_train, cv= kflod) # model: 5번 * score 5번 = 25번
    print('교차검증점수 : ', score)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))


'''
교차검증점수 :  [0.48675134 0.42825909 0.32825119 0.33361883 0.45601166]
'''
