# allAlgorithmsdp kflod넣기

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_iris()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) #shuffle: 행을 섞는것

allAlgorithms = all_estimators(type_filter='classifier') #all_estimators 추정지 추정값

for (name, algorithm) in allAlgorithms:  #allAlgorithms에서 인자 2개(name, algorithm)
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold) #cv에 나누고 싶은 수 입력해도 괜춘
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 :', scores) # 답 5개씩 나온다
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

'''
전처리만
AdaBoostClassifier 의 정답률 : 0.9666666666666667
BaggingClassifier 의 정답률 : 0.9
BernoulliNB 의 정답률 : 0.3
CalibratedClassifierCV 의 정답률 : 0.9333333333333333
CategoricalNB 의 정답률 : 0.9
CheckingClassifier 의 정답률 : 0.3
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : 0.7
DecisionTreeClassifier 의 정답률 : 0.8666666666666667
DummyClassifier 의 정답률 : 0.3333333333333333
ExtraTreeClassifier 의 정답률 : 0.9333333333333333
ExtraTreesClassifier 의 정답률 : 0.9333333333333333
GaussianNB 의 정답률 : 0.9333333333333333
GaussianProcessClassifier 의 정답률 : 0.9666666666666667
GradientBoostingClassifier 의 정답률 : 0.9666666666666667
HistGradientBoostingClassifier 의 정답률 : 0.9666666666666667
KNeighborsClassifier 의 정답률 : 0.9666666666666667
LabelPropagation 의 정답률 : 0.9666666666666667
LabelSpreading 의 정답률 : 0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 : 1.0
LinearSVC 의 정답률 : 0.9666666666666667
LogisticRegression 의 정답률 : 0.9666666666666667
LogisticRegressionCV 의 정답률 : 0.9666666666666667
MLPClassifier 의 정답률 : 1.0
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : 0.8666666666666667
NearestCentroid 의 정답률 : 0.9
NuSVC 의 정답률 : 0.9666666666666667
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : 0.8
Perceptron 의 정답률 : 0.7333333333333333
QuadraticDiscriminantAnalysis 의 정답률 : 1.0
RadiusNeighborsClassifier 의 정답률 : 0.9333333333333333
RandomForestClassifier 의 정답률 : 0.9666666666666667
RidgeClassifier 의 정답률 : 0.8333333333333334
RidgeClassifierCV 의 정답률 : 0.8333333333333334
SGDClassifier 의 정답률 : 0.9
SVC 의 정답률 : 0.9666666666666667
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
============================================================================================
kflod썼을때
AdaBoostClassifier 의 정답률 : [0.875      1.         0.95833333 0.83333333 0.91666667]
BaggingClassifier 의 정답률 : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333]
BernoulliNB 의 정답률 : [0.29166667 0.29166667 0.16666667 0.33333333 0.29166667]
CalibratedClassifierCV 의 정답률 : [0.95833333 0.875      0.875      0.79166667 0.95833333]
CategoricalNB 의 정답률 : [0.875      1.         1.         0.875      0.95833333]
CheckingClassifier 의 정답률 : [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : [0.75       0.54166667 0.625      0.70833333 0.66666667]
DecisionTreeClassifier 의 정답률 : [0.91666667 0.91666667 0.91666667 0.95833333 1.        ]
DummyClassifier 의 정답률 : [0.33333333 0.20833333 0.33333333 0.29166667 0.375     ]
ExtraTreeClassifier 의 정답률 : [0.875      0.83333333 0.95833333 0.95833333 0.91666667]
ExtraTreesClassifier 의 정답률 : [0.95833333 0.91666667 0.91666667 0.95833333 0.95833333]
GaussianNB 의 정답률 : [1.         1.         0.95833333 0.91666667 0.95833333]
GaussianProcessClassifier 의 정답률 : [0.91666667 0.95833333 1.         0.91666667 1.        ]
GradientBoostingClassifier 의 정답률 : [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667]
HistGradientBoostingClassifier 의 정답률 : [0.875      1.         0.95833333 0.875      0.95833333]
KNeighborsClassifier 의 정답률 : [1.         0.95833333 0.95833333 1.         0.91666667]
LabelPropagation 의 정답률 : [0.95833333 0.91666667 0.95833333 0.95833333 0.95833333]
LabelSpreading 의 정답률 : [1.         0.95833333 0.83333333 1.         1.        ]
LinearDiscriminantAnalysis 의 정답률 : [1.         1.         1.         0.95833333 0.95833333]
LinearSVC 의 정답률 : [0.83333333 1.         0.95833333 1.         0.95833333]
LogisticRegression 의 정답률 : [0.95833333 0.95833333 0.95833333 0.83333333 1.        ]
LogisticRegressionCV 의 정답률 : [0.95833333 1.         0.91666667 0.875      0.91666667]
MLPClassifier 의 정답률 : [1.         1.         1.         0.95833333 0.91666667]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : [0.95833333 0.45833333 0.95833333 0.83333333 0.875     ]
NearestCentroid 의 정답률 : [0.875      0.95833333 1.         0.95833333 0.875     ]
NuSVC 의 정답률 : [0.875      0.91666667 1.         0.91666667 0.95833333]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : [0.83333333 0.875      0.75       0.95833333 1.        ]
Perceptron 의 정답률 : [0.875      0.41666667 0.75       0.45833333 0.83333333]
QuadraticDiscriminantAnalysis 의 정답률 : [0.95833333 1.         0.95833333 0.95833333 0.95833333]
RadiusNeighborsClassifier 의 정답률 : [0.95833333 1.         0.91666667 0.875      1.        ]
RandomForestClassifier 의 정답률 : [1.         0.91666667 0.95833333 0.95833333 0.95833333]
RidgeClassifier 의 정답률 : [0.75       0.91666667 0.625      0.83333333 0.875     ]
RidgeClassifierCV 의 정답률 : [0.91666667 0.91666667 0.79166667 0.875      0.79166667]
SGDClassifier 의 정답률 : [0.95833333 0.75       0.875      0.875      0.5       ]
SVC 의 정답률 : [0.95833333 0.95833333 0.95833333 0.95833333 0.875     ]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
