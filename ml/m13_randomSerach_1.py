# Grid로 촘촘하게 데이터 검사 한 번에 돌리기(함수 여러개 입력)
# 파라미터 값 높은 것을 찾는 것이 목표다(best parameter)
# Grid보다 더 빠르게 값을 추출하는 방법 RandomizedSearchCV

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_iris()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=77, shuffle=True, train_size=0.8
) #이렇게 하면 train에 있는것을 kflod에서 5로 나누게 된다.(val이 생성된 셈이다.)

kflod = KFold(n_splits=5, shuffle=True) #데이터를 5씩 잘라서 model과 연결

parameters = [
    {"C" : [1, 10, 100, 1000], 'kernel':['linear']},#4번
    {"C" : [1, 10, 100], 'kernel':['ref'], 'gamma':[0.001, 0.0001]},#6번
    {"C" : [1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}#8번
] #총 18번을 n_splits=5로 18*5 =90번 돈다(이것은 우리가 정하는 것) ---> 랜덤하게 하므로 안넣은 값은 디폴트됨

# --------------------------------------------------------------------------------------------
#  2. 모델링

# model = SVC()
 # '최근접 이웃':KNeighborsClassifier(),
 #    '로지스틱 회귀':LogisticRegression(),
 #   '비선형 SVM':SVC(),
 #   '결정 트리':DecisionTreeClassifier(),
 #   '랜덤 포레스트':RandomForestClassifier()}

model = RandomizedSearchCV(SVC(), parameters, cv=kflod) #데이터를 모두 감싸고 90번 돈다

# 3. 훈련 =====================================================================
model.fit(x_train,y_train)

#4. 평가 ==================================================================
print('최적의 매개변수: ', model.best_estimator_) #가장 좋은 성능을 보인 모델이 반환

y_pred = model.predict(x_test) #90번 돈것중에 가장 좋은 것으로 반환한다.
print('최종정답률', accuracy_score(y_test, y_pred))

result = model.score(x_test, y_test) #evaluate 대신 score사용
print('result:', result) 

'''
for 문으로 반복해 측정해서 나온다.
분류형 모델 전체를 sklearn의 all_estimators 여기다 저장해 놓은것
이중 젤 좋은것 1.0 제외
AdaBoostClassifier 의 정답률 : 0.9666666666666667
tensorflow
keras33과 비교
0.9666666388511658
GridSearchCV
최적의 매개변수:  SVC(C=1, kernel='linear')
최종정답률 0.8666666666666667
result: 0.8666666666666667
---> 자동으로 튠 잡아준다
RandomizedSearchCV
최적의 매개변수:  SVC(C=1000, gamma=0.001, kernel='sigmoid')
최종정답률 0.8666666666666667
result: 0.8666666666666667
'''
