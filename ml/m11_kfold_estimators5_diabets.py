#m 09_selectModel 4,5은 형식이 다르다
## 회귀형식

from sklearn.datasets import load_iris, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) #shuffle: 행을 섞는것, train에서 5개로 나누는것(val이 여기서 생성)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:  #allAlgorithms에서 인자 2개(name, algorithm)
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 :', scores)
    except: #예외인경우
        #continue
        print(name, '은 없는 놈!') #이렇게 하면 안끊기고 반복되 다 나온다


'''
for 문으로 반복해 측정해서 나온다.
분류형 모델 전체를 sklearn의 all_estimators 여기다 저장해 놓은것
이중 젤 좋은것 1.0 제외
ARDRegression 의 정답률 : 0.5278342233069477
---------------------------
tensorflow
keras54과 비교
R2 : 0.5754409282584073
'''

'''
전처리만
ARDRegression 의 정답률 : 0.5278342233068394
AdaBoostRegressor 의 정답률 : 0.41488987363586416
BaggingRegressor 의 정답률 : 0.39241036714091193
BayesianRidge 의 정답률 : 0.5193410135537663
CCA 의 정답률 : 0.48879618038824757
DecisionTreeRegressor 의 정답률 : -0.2630110967089716
DummyRegressor 의 정답률 : -0.07457975637038539
ElasticNet 의 정답률 : -0.06518000443720706
ElasticNetCV 의 정답률 : 0.4294375480398558
ExtraTreeRegressor 의 정답률 : -0.0919627121599278
ExtraTreesRegressor 의 정답률 : 0.44881100855981804
GammaRegressor 의 정답률 : -0.06869757267027454
GaussianProcessRegressor 의 정답률 : -16.57366391984241
GeneralizedLinearRegressor 의 정답률 : -0.06771406705799343
GradientBoostingRegressor 의 정답률 : 0.3633006839036781
HistGradientBoostingRegressor 의 정답률 : 0.3504135950167052
HuberRegressor 의 정답률 : 0.5205018285661304
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 : 0.35838503635518537
KernelRidge 의 정답률 : -4.4187445504449405
Lars 의 정답률 : 0.21479550446394002
LarsCV 의 정답률 : 0.516365352104498
Lasso 의 정답률 : 0.33086319953362164
LassoCV 의 정답률 : 0.5222186221789182
LassoLars 의 정답률 : 0.3570808988866827
LassoLarsCV 의 정답률 : 0.5214536844628463
LassoLarsIC 의 정답률 : 0.5224736703335271
LinearRegression 의 정답률 : 0.525204262124852
LinearSVR 의 정답률 : -0.8306231508702273
MLPRegressor 의 정답률 : -4.078359694300735
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 : 0.07746639731663862
OrthogonalMatchingPursuit 의 정답률 : 0.3337053538857254
OrthogonalMatchingPursuitCV 의 정답률 : 0.5257611661032995
PLSCanonical 의 정답률 : -1.2663831979876923
PLSRegression 의 정답률 : 0.5042012880276586
PassiveAggressiveRegressor 의 정답률 : 0.45931208589361017
PoissonRegressor 의 정답률 : 0.29880208432725275
RANSACRegressor 의 정답률 : 0.16047234414972678
RadiusNeighborsRegressor 의 정답률 : -0.07457975637038539
RandomForestRegressor 의 정답률 : 0.4355269046206657
RegressorChain 은 없는 놈!
Ridge 의 정답률 : 0.40179727975154844
RidgeCV 의 정답률 : 0.5132298404989653
SGDRegressor 의 정답률 : 0.37698096451536955
SVR 의 정답률 : 0.008054881772852074
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 : 0.515688511490622
TransformedTargetRegressor 의 정답률 : 0.525204262124852
TweedieRegressor 의 정답률 : -0.06771406705799343
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
===============================================================
Kflod했을 때
ARDRegression 의 정답률 : [0.46886441 0.53577134 0.48039656 0.47051539 0.42372317]
AdaBoostRegressor 의 정답률 : [0.34913775 0.40294079 0.50944196 0.41940452 0.39273228]
BaggingRegressor 의 정답률 : [0.52640117 0.26852565 0.27815285 0.40899637 0.3565601 ]
BayesianRidge 의 정답률 : [0.38541159 0.47365281 0.46776747 0.52010877 0.45124652]
CCA 의 정답률 : [0.51336967 0.43331248 0.45726108 0.27193794 0.34117485]
DecisionTreeRegressor 의 정답률 : [-0.16175803  0.09041561 -0.22951626 -0.42428264  0.25359399]
DummyRegressor 의 정답률 : [-0.00731634 -0.05520356 -0.00500472 -0.02574646 -0.00759932]
ElasticNet 의 정답률 : [-0.01032309 -0.03578684 -0.00777263  0.00931139  0.0097532 ]
ElasticNetCV 의 정답률 : [0.43581958 0.47957599 0.48163164 0.43412051 0.32544066]
ExtraTreeRegressor 의 정답률 : [-0.21517771  0.04969123  0.09155992  0.19994583 -0.11161937]
ExtraTreesRegressor 의 정답률 : [0.42210695 0.45793435 0.29500182 0.39116856 0.37095566]
GammaRegressor 의 정답률 : [ 0.00616509  0.00605374  0.00461374  0.00358291 -0.00395125]
GaussianProcessRegressor 의 정답률 : [-13.33964089  -8.94405934  -8.40325125 -16.23209615  -9.08282259]
GeneralizedLinearRegressor 의 정답률 : [ 0.00475772  0.00633758 -0.00179716 -0.02679528 -0.06252541]
GradientBoostingRegressor 의 정답률 : [0.16393243 0.49123684 0.35204107 0.24308891 0.6208567 ]
HistGradientBoostingRegressor 의 정답률 : [0.00733287 0.34252367 0.18872233 0.60937658 0.49370174]
HuberRegressor 의 정답률 : [0.51695363 0.29685388 0.37394553 0.46933634 0.49249937]
IsotonicRegression 의 정답률 : [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : [0.23552089 0.40264345 0.2933146  0.38845078 0.50902546]
KernelRidge 의 정답률 : [-3.74650221 -3.75164558 -3.61028344 -3.13827328 -3.41282111]
Lars 의 정답률 : [0.44777559 0.56367525 0.38270718 0.37256482 0.5024898 ]
LarsCV 의 정답률 : [0.54253108 0.48294939 0.31568785 0.55386286 0.39645583]
Lasso 의 정답률 : [0.37002138 0.33064461 0.34001581 0.28453074 0.31501396]
LassoCV 의 정답률 : [0.50082874 0.54200865 0.52509342 0.37779682 0.39293973]
LassoLars 의 정답률 : [0.401309   0.38252914 0.38110976 0.41034187 0.34088281]
LassoLarsCV 의 정답률 : [0.38298625 0.34590167 0.51688032 0.56661359 0.49555603]
LassoLarsIC 의 정답률 : [0.41568946 0.54463481 0.42712961 0.46405307 0.47431895]
LinearRegression 의 정답률 : [0.44009367 0.40883759 0.38534231 0.63351746 0.37888162]
LinearSVR 의 정답률 : [-0.66942358 -0.60638552 -0.39615157 -0.32519425 -0.30651828]
MLPRegressor 의 정답률 : [-2.32805872 -2.63235237 -2.92939991 -2.97974268 -3.06473846]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 : [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 : [nan nan nan nan nan]
MultiTaskLasso 의 정답률 : [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 : [nan nan nan nan nan]
NuSVR 의 정답률 : [0.15372296 0.14101083 0.11927729 0.09487846 0.10437987]
OrthogonalMatchingPursuit 의 정답률 : [0.24019685 0.39424111 0.30278027 0.19862909 0.19969614]
OrthogonalMatchingPursuitCV 의 정답률 : [0.58184721 0.44384817 0.3791262  0.43691983 0.48861714]
PLSCanonical 의 정답률 : [-1.96899642 -0.74075552 -1.41295066 -1.31903957 -1.63152358]
PLSRegression 의 정답률 : [0.31546868 0.49504191 0.57738563 0.52446385 0.44123199]
PassiveAggressiveRegressor 의 정답률 : [0.54350673 0.48379744 0.44017467 0.17487547 0.37338167]
PoissonRegressor 의 정답률 : [0.37089511 0.28432992 0.29569114 0.28001503 0.32085275]
RANSACRegressor 의 정답률 : [-0.14180757 -0.10537699  0.19772008  0.23423016  0.37876041]
RadiusNeighborsRegressor 의 정답률 : [-0.00330172 -0.03219846 -0.00147603 -0.00059252 -0.00368371]
RandomForestRegressor 의 정답률 : [0.57140753 0.41335391 0.41384276 0.2280489  0.38608769]
RegressorChain 은 없는 놈!
Ridge 의 정답률 : [0.34947202 0.43618581 0.40048309 0.32172683 0.45097445]
RidgeCV 의 정답률 : [0.40190408 0.44474816 0.5204815  0.48693889 0.45437967]
SGDRegressor 의 정답률 : [0.35442057 0.27347471 0.46573222 0.33936382 0.38056135]
SVR 의 정답률 : [0.13146497 0.06806565 0.08135758 0.14996464 0.14768609]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 : [0.38696755 0.40789584 0.52967892 0.33642497 0.50714672]
TransformedTargetRegressor 의 정답률 : [0.39641874 0.53790605 0.44985516 0.47545804 0.48450929]
TweedieRegressor 의 정답률 : [-0.02640039 -0.05320412  0.00586698  0.00450986 -0.00477393]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답률 : [nan nan nan nan nan]
'''
