#m 09_selectModel 4,5은 형식이 다르다
## 회귀형식

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_boston()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:  #allAlgorithms에서 인자 2개(name, algorithm)
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 :', r2_score(y_test, y_pred))
    except: #예외인경우
        #continue
        print(name, '은 없는 놈!') #이렇게 하면 안끊기고 반복되 다 나온다

#회귀여서 바꾼것 regressor, r2_score

'''
for 문으로 반복해 측정해서 나온다.
분류형 모델 전체를 sklearn의 all_estimators 여기다 저장해 놓은것
이중 젤 좋은것 1.0 제외
HistGradientBoostingRegressor 의 정답률 : 0.8991491407747458

---------------------------
tensorflow
keras54과 비교
R2 :  0.8125122028343794

'''

