#m09_selectModel 1~3까지는 같다 

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_wine()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) #shuffle: 행을 섞는것

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:  #allAlgorithms에서 인자 2개(name, algorithm)
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 :', scores)
    except: #예외인경우
        #continue
        print(name, '은 없는 놈!')

'''
for 문으로 반복해 측정해서 나온다.
분류형 모델 전체를 sklearn의 all_estimators 여기다 저장해 놓은것
이중 젤 좋은것 1.0 제외
CalibratedClassifierCV 의 정답률 : 0.9444444444444444
---------------------------
tensorflow
keras33과 비교
0.9722222089767456
'''
'''
전처리만
AdaBoostClassifier 의 정답률 : 0.8888888888888888
BaggingClassifier 의 정답률 : 0.8888888888888888
BernoulliNB 의 정답률 : 0.4166666666666667
CalibratedClassifierCV 의 정답률 : 0.9444444444444444
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 : 0.3888888888888889
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : 0.6388888888888888
DecisionTreeClassifier 의 정답률 : 0.8888888888888888
DummyClassifier 의 정답률 : 0.3055555555555556
ExtraTreeClassifier 의 정답률 : 0.8888888888888888
ExtraTreesClassifier 의 정답률 : 0.9722222222222222
GaussianNB 의 정답률 : 0.9166666666666666
GaussianProcessClassifier 의 정답률 : 0.3888888888888889
GradientBoostingClassifier 의 정답률 : 0.9166666666666666
HistGradientBoostingClassifier 의 정답률 : 0.9444444444444444
KNeighborsClassifier 의 정답률 : 0.6944444444444444
LabelPropagation 의 정답률 : 0.5833333333333334
LabelSpreading 의 정답률 : 0.5833333333333334
LinearDiscriminantAnalysis 의 정답률 : 0.9722222222222222
LinearSVC 의 정답률 : 0.9166666666666666
LogisticRegression 의 정답률 : 0.9444444444444444
LogisticRegressionCV 의 정답률 : 0.8888888888888888
MLPClassifier 의 정답률 : 0.5277777777777778
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : 0.7777777777777778
NearestCentroid 의 정답률 : 0.6388888888888888
NuSVC 의 정답률 : 0.8611111111111112
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : 0.2222222222222222
Perceptron 의 정답률 : 0.6111111111111112
QuadraticDiscriminantAnalysis 의 정답률 : 1.0
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 : 0.9444444444444444
RidgeClassifier 의 정답률 : 0.9444444444444444
RidgeClassifierCV 의 정답률 : 0.9444444444444444
SGDClassifier 의 정답률 : 0.4722222222222222
SVC 의 정답률 : 0.6111111111111112
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
===================================================================
Kflod 했을 때
AdaBoostClassifier 의 정답률 : [0.96551724 0.82758621 0.96428571 0.85714286 0.89285714]
BaggingClassifier 의 정답률 : [0.96551724 0.93103448 0.96428571 1.         1.        ]
BernoulliNB 의 정답률 : [0.03448276 0.27586207 0.21428571 0.42857143 0.35714286]
CalibratedClassifierCV 의 정답률 : [0.96551724 0.86206897 0.92857143 0.82142857 1.        ]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 : [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : [0.68965517 0.72413793 0.53571429 0.64285714 0.64285714]
DecisionTreeClassifier 의 정답률 : [0.86206897 0.86206897 0.89285714 0.82142857 0.96428571]
DummyClassifier 의 정답률 : [0.34482759 0.24137931 0.39285714 0.32142857 0.35714286]
ExtraTreeClassifier 의 정답률 : [0.82758621 0.86206897 0.82142857 0.75       0.92857143]
ExtraTreesClassifier 의 정답률 : [0.93103448 1.         1.         1.         1.        ]
GaussianNB 의 정답률 : [0.96551724 1.         0.96428571 0.96428571 0.96428571]
GaussianProcessClassifier 의 정답률 : [0.48275862 0.34482759 0.53571429 0.35714286 0.64285714]
GradientBoostingClassifier 의 정답률 : [0.93103448 0.96551724 0.89285714 1.         0.85714286]
HistGradientBoostingClassifier 의 정답률 : [0.96551724 1.         0.89285714 0.96428571 1.        ]
KNeighborsClassifier 의 정답률 : [0.68965517 0.68965517 0.71428571 0.75       0.82142857]
LabelPropagation 의 정답률 : [0.34482759 0.27586207 0.46428571 0.46428571 0.60714286]
LabelSpreading 의 정답률 : [0.27586207 0.51724138 0.46428571 0.35714286 0.53571429]
LinearDiscriminantAnalysis 의 정답률 : [1.         0.96551724 0.96428571 0.96428571 0.96428571]
LinearSVC 의 정답률 : [0.82758621 0.86206897 0.67857143 0.92857143 0.75      ]
LogisticRegression 의 정답률 : [1.         0.93103448 0.92857143 0.78571429 1.        ]
LogisticRegressionCV 의 정답률 : [0.89655172 0.93103448 0.92857143 1.         0.96428571]
MLPClassifier 의 정답률 : [0.65517241 0.51724138 0.46428571 0.89285714 0.39285714]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : [0.86206897 0.86206897 0.92857143 0.71428571 0.78571429]
NearestCentroid 의 정답률 : [0.89655172 0.65517241 0.75       0.71428571 0.75      ]
NuSVC 의 정답률 : [0.86206897 0.93103448 0.85714286 0.92857143 0.82142857]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : [0.65517241 0.5862069  0.25       0.46428571 0.42857143]
Perceptron 의 정답률 : [0.65517241 0.4137931  0.75       0.5        0.5       ]
QuadraticDiscriminantAnalysis 의 정답률 : [0.93103448 0.93103448 0.96428571 1.         1.        ]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 : [1.         1.         1.         1.         0.96428571]
RidgeClassifier 의 정답률 : [1.         0.96551724 1.         1.         0.96428571]
RidgeClassifierCV 의 정답률 : [0.96551724 0.96551724 0.96428571 1.         1.        ]
SGDClassifier 의 정답률 : [0.34482759 0.72413793 0.67857143 0.60714286 0.60714286]
SVC 의 정답률 : [0.68965517 0.75862069 0.64285714 0.64285714 0.57142857]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
