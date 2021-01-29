### RandomizedSearchCV iris~diabet하기
#RandomizedSearchCV 뜻 알고 Randomized에 적용되는 random이 몇퍼센트인지 알아보기
# RandomForest로 하고 time도 달아서 얼마나 걸리는지 보기

# 모델 : RandomForestClassifier #선생님은 왜 이것만?
# 파라미터 값 높은 것을 찾는 것이 목표다(best parameter)

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
import timeit# 시간측정
import pandas as pd
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_wine()
x= dataset.data
y= dataset.target
print(x.shape, y.shape) #(178, 13) (178,)
# dataset = load_iris()
# x= dataset.data
# y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True)


parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth': [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1]} #총15번
]
 
#--------------------------------------------------------------------------------
start_time = timeit.default_timer() # 시작 시간 체크
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kflod)

# 3. 훈련 =====================================================================
model.fit(x_train,y_train)

#4. 평가 ==================================================================
print('최적의 매개변수: ', model.best_estimator_) #가장 좋은 성능을 보인 모델이 반환

y_pred = model.predict(x_test) #90번 돈것중에 가장 좋은 것으로 반환한다.
print('최종정답률', r2_score(y_test, y_pred))

result = model.score(x_test, y_test) #evaluate 대신 score사용
print('result:', result) 

terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))

'''
GridSearchCV
최적의 매개변수:  RandomForestRegressor(n_estimators=200)
최종정답률 0.8959186428479842
RandomizedSearchCV
최적의 매개변수:  RandomForestClassifier(n_jobs=-1)
최종정답률 1.0
result: 1.0
7.561711초 걸렸습니다.
'''
