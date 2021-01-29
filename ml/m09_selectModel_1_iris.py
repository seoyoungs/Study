from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_iris()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier') #$all_estimators 추정치

for (name, algorithm) in allAlgorithms:  #allAlgorithms에서 인자 2개(name, algorithm)
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 :', accuracy_score(y_test, y_pred))
    except: #예외인경우
        #continue
        print(name, '은 없는 놈!') #이렇게 하면 안끊기고 반복되 다 나온다

#이렇게 하면 iris에서 sklearn버전 0.23.2에서 볼수 있는 모델링 다 볼 수 있다

# import sklearn #sklearn버전 확인
# print(sklearn.__version__) #0.23.2 근데 모델링 다하려면 22이하여야함
#그럼 어떻게 다 나오게 할까? for문에 예외경우(except)를 쓴다.


'''
for 문으로 반복해 측정해서 나온다.
분류형 모델 전체를 sklearn의 all_estimators 여기다 저장해 놓은것
이중 젤 좋은것 1.0 제외
AdaBoostClassifier 의 정답률 : 0.9666666666666667

tensorflow
keras33과 비교
0.9666666388511658
'''

