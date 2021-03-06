# 모델 : RandomForestClassifier #선생님은 왜 이것만?
# 회귀는 RandomForestRegressor
# 파라미터 값 높은 것을 찾는 것이 목표다(best parameter)

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True)


parameters = [
    {'n_estimators' : [100,200], 'max_depth': [6,8,10,12],'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split': [2,3,5,10],'n_jobs':[-1,2,4]} #총15번*5 =75, n_jobs':[-1]--> 코어전부 쓰겠다. n_jobs':[2]-두개만 쓰겠다
] #이렇게 합쳐서 쓸 수 도 있다.

'''
parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth': [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1]} #총15번*5 =75
]
'''

model = GridSearchCV(RandomForestRegressor(), parameters, cv=kflod)

# 3. 훈련 =====================================================================
model.fit(x_train,y_train)

#4. 평가 ==================================================================
print('최적의 매개변수: ', model.best_estimator_) #가장 좋은 성능을 보인 모델이 반환

y_pred = model.predict(x_test) #90번 돈것중에 가장 좋은 것으로 반환한다.
print('최종정답률', r2_score(y_test, y_pred))

'''
'n_jobs':[-1]
최적의 매개변수:  RandomForestRegressor(min_samples_leaf=7)
최종정답률 0.4858531031411961
'n_jobs':[-1,2,4]
최적의 매개변수:  RandomForestRegressor(min_samples_leaf=5)
최종정답률 0.47570980698955323
최적의 매개변수:  RandomForestRegressor(max_depth=8, min_samples_leaf=5)
최종정답률 0.4768387619214587
'''
