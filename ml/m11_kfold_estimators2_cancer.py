from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_breast_cancer()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

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
AdaBoostClassifier 의 정답률 : 0.9736842105263158
tensorflow
keras33과 비교
0.9824561476707458
'''
'''
전처리만 한것
AdaBoostClassifier 의 정답률 : 0.9736842105263158
BaggingClassifier 의 정답률 : 0.9649122807017544
BernoulliNB 의 정답률 : 0.6578947368421053
CalibratedClassifierCV 의 정답률 : 0.9824561403508771
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 : 0.34210526315789475
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : 0.9473684210526315
DecisionTreeClassifier 의 정답률 : 0.9473684210526315
DummyClassifier 의 정답률 : 0.49122807017543857
ExtraTreeClassifier 의 정답률 : 0.956140350877193
ExtraTreesClassifier 의 정답률 : 0.9736842105263158
GaussianNB 의 정답률 : 0.9736842105263158
GaussianProcessClassifier 의 정답률 : 0.9298245614035088
GradientBoostingClassifier 의 정답률 : 0.9736842105263158
HistGradientBoostingClassifier 의 정답률 : 0.9736842105263158
KNeighborsClassifier 의 정답률 : 0.956140350877193
LabelPropagation 의 정답률 : 0.3684210526315789
LabelSpreading 의 정답률 : 0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 : 0.9912280701754386
LinearSVC 의 정답률 : 0.9210526315789473
LogisticRegression 의 정답률 : 0.9736842105263158
LogisticRegressionCV 의 정답률 : 0.9736842105263158
MLPClassifier 의 정답률 : 0.9649122807017544
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : 0.9473684210526315
NearestCentroid 의 정답률 : 0.9298245614035088
NuSVC 의 정답률 : 0.9385964912280702
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : 0.9385964912280702
Perceptron 의 정답률 : 0.8421052631578947
QuadraticDiscriminantAnalysis 의 정답률 : 0.9649122807017544
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 : 0.9649122807017544
RidgeClassifier 의 정답률 : 0.9824561403508771
RidgeClassifierCV 의 정답률 : 0.9824561403508771
SGDClassifier 의 정답률 : 0.9122807017543859
SVC 의 정답률 : 0.956140350877193
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
'''
전처리만 한것
AdaBoostClassifier 의 정답률 : 0.9736842105263158
BaggingClassifier 의 정답률 : 0.9649122807017544
BernoulliNB 의 정답률 : 0.6578947368421053
CalibratedClassifierCV 의 정답률 : 0.9824561403508771
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 : 0.34210526315789475
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : 0.9473684210526315
DecisionTreeClassifier 의 정답률 : 0.9473684210526315
DummyClassifier 의 정답률 : 0.49122807017543857
ExtraTreeClassifier 의 정답률 : 0.956140350877193
ExtraTreesClassifier 의 정답률 : 0.9736842105263158
GaussianNB 의 정답률 : 0.9736842105263158
GaussianProcessClassifier 의 정답률 : 0.9298245614035088
GradientBoostingClassifier 의 정답률 : 0.9736842105263158
HistGradientBoostingClassifier 의 정답률 : 0.9736842105263158
KNeighborsClassifier 의 정답률 : 0.956140350877193
LabelPropagation 의 정답률 : 0.3684210526315789
LabelSpreading 의 정답률 : 0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 : 0.9912280701754386
LinearSVC 의 정답률 : 0.9210526315789473
LogisticRegression 의 정답률 : 0.9736842105263158
LogisticRegressionCV 의 정답률 : 0.9736842105263158
MLPClassifier 의 정답률 : 0.9649122807017544
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : 0.9473684210526315
NearestCentroid 의 정답률 : 0.9298245614035088
NuSVC 의 정답률 : 0.9385964912280702
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : 0.9385964912280702
Perceptron 의 정답률 : 0.8421052631578947
QuadraticDiscriminantAnalysis 의 정답률 : 0.9649122807017544
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 : 0.9649122807017544
RidgeClassifier 의 정답률 : 0.9824561403508771
RidgeClassifierCV 의 정답률 : 0.9824561403508771
SGDClassifier 의 정답률 : 0.9122807017543859
SVC 의 정답률 : 0.956140350877193
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
=========================================================================================
Kflod 한것
AdaBoostClassifier 의 정답률 : [0.93406593 0.95604396 0.96703297 0.97802198 0.94505495]
BaggingClassifier 의 정답률 : [0.96703297 0.93406593 0.92307692 0.9010989  0.96703297]
BernoulliNB 의 정답률 : [0.76923077 0.54945055 0.54945055 0.62637363 0.6043956 ]
CalibratedClassifierCV 의 정답률 : [0.92307692 0.92307692 0.92307692 0.87912088 0.9010989 ]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 : [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 : [0.9010989  0.85714286 0.87912088 0.89010989 0.89010989]
DecisionTreeClassifier 의 정답률 : [0.94505495 0.94505495 0.91208791 0.9010989  0.91208791]
DummyClassifier 의 정답률 : [0.47252747 0.57142857 0.46153846 0.46153846 0.46153846]
ExtraTreeClassifier 의 정답률 : [0.93406593 0.91208791 0.91208791 0.93406593 0.9010989 ]
ExtraTreesClassifier 의 정답률 : [0.95604396 0.98901099 0.9010989  0.96703297 0.97802198]
GaussianNB 의 정답률 : [0.94505495 0.94505495 0.91208791 0.93406593 0.92307692]
GaussianProcessClassifier 의 정답률 : [0.89010989 0.94505495 0.9010989  0.9010989  0.89010989]
GradientBoostingClassifier 의 정답률 : [0.95604396 0.98901099 0.94505495 0.93406593 0.93406593]
HistGradientBoostingClassifier 의 정답률 : [0.94505495 0.95604396 0.95604396 0.98901099 0.93406593]
KNeighborsClassifier 의 정답률 : [0.93406593 0.91208791 0.9010989  0.92307692 0.94505495]
LabelPropagation 의 정답률 : [0.43956044 0.30769231 0.43956044 0.41758242 0.40659341]
LabelSpreading 의 정답률 : [0.41758242 0.41758242 0.40659341 0.27472527 0.47252747]
LinearDiscriminantAnalysis 의 정답률 : [0.93406593 0.98901099 0.95604396 0.96703297 0.92307692]
LinearSVC 의 정답률 : [0.9010989  0.93406593 0.87912088 0.92307692 0.91208791]
LogisticRegression 의 정답률 : [0.89010989 0.91208791 0.96703297 0.92307692 0.94505495]
LogisticRegressionCV 의 정답률 : [0.96703297 0.95604396 0.95604396 0.96703297 0.92307692]
MLPClassifier 의 정답률 : [0.97802198 0.9010989  0.93406593 0.89010989 0.94505495]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 : [0.87912088 0.91208791 0.83516484 0.89010989 0.9010989 ]
NearestCentroid 의 정답률 : [0.89010989 0.86813187 0.83516484 0.86813187 0.95604396]
NuSVC 의 정답률 : [0.91208791 0.84615385 0.84615385 0.87912088 0.86813187]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 : [0.93406593 0.85714286 0.91208791 0.82417582 0.87912088]
Perceptron 의 정답률 : [0.94505495 0.68131868 0.87912088 0.94505495 0.86813187]
QuadraticDiscriminantAnalysis 의 정답률 : [0.97802198 0.95604396 0.92307692 0.96703297 0.94505495]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 : [0.98901099 0.92307692 0.95604396 0.92307692 0.97802198]
RidgeClassifier 의 정답률 : [0.92307692 0.97802198 0.94505495 0.96703297 0.94505495]
RidgeClassifierCV 의 정답률 : [0.97802198 0.96703297 0.96703297 0.92307692 0.94505495]
SGDClassifier 의 정답률 : [0.82417582 0.79120879 0.82417582 0.92307692 0.9010989 ]
SVC 의 정답률 : [0.9010989  0.91208791 0.93406593 0.9010989  0.9010989 ]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
