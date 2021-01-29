#m 09_selectModel 4,5은 형식이 다르다
## 회귀형식

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_boston()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                      test_size = 0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) #shuffle: 행을 섞는것

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

'''
전처리했을 때
ARDRegression 의 정답률 : 0.7512651671065581
AdaBoostRegressor 의 정답률 : 0.839145014916421
BaggingRegressor 의 정답률 : 0.8878088485039545
BayesianRidge 의 정답률 : 0.7444785336818114
CCA 의 정답률 : 0.7270542664211517
DecisionTreeRegressor 의 정답률 : 0.8325323478327442
DummyRegressor 의 정답률 : -0.0007982049217318821
ElasticNet 의 정답률 : 0.6990500898755508
ElasticNetCV 의 정답률 : 0.6902681369495264
ExtraTreeRegressor 의 정답률 : 0.7576353029098427
ExtraTreesRegressor 의 정답률 : 0.9017907910889836
GammaRegressor 의 정답률 : -0.0007982049217318821
GaussianProcessRegressor 의 정답률 : -5.639147690233129
GeneralizedLinearRegressor 의 정답률 : 0.6917874063129013
GradientBoostingRegressor 의 정답률 : 0.8959093461503634
HistGradientBoostingRegressor 의 정답률 : 0.8991491407747458
HuberRegressor 의 정답률 : 0.7233379135400204
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 : 0.6390759816821279
KernelRidge 의 정답률 : 0.7744886782300767
Lars 의 정답률 : 0.7521800808693164
LarsCV 의 정답률 : 0.7570138649983484
Lasso 의 정답률 : 0.6855879495660049
LassoCV 의 정답률 : 0.7154057460487299
LassoLars 의 정답률 : -0.0007982049217318821
LassoLarsCV 의 정답률 : 0.7570138649983484
LassoLarsIC 의 정답률 : 0.754094595988446
LinearRegression 의 정답률 : 0.7521800808693141
LinearSVR 의 정답률 : 0.34215585918258806
MLPRegressor 의 정답률 : 0.5127819453839477
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 : 0.32534704254368274
OrthogonalMatchingPursuit 의 정답률 : 0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률 : 0.7377665753906506
PLSCanonical 의 정답률 : -1.7155095545127699
PLSRegression 의 정답률 : 0.7666940310402938
PassiveAggressiveRegressor 의 정답률 : -0.3531689316826272
PoissonRegressor 의 정답률 : 0.8014250117852569
RANSACRegressor 의 정답률 : 0.5303752403957808
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 : 0.8914842871870071
RegressorChain 은 없는 놈!
Ridge 의 정답률 : 0.7539303499010775
RidgeCV 의 정답률 : 0.7530092298810112
SGDRegressor 의 정답률 : -8.253528206923099e+26
SVR 의 정답률 : 0.2868662719877668
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 : 0.7943020571701422
TransformedTargetRegressor 의 정답률 : 0.7521800808693141
TweedieRegressor 의 정답률 : 0.6917874063129013
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
==================================================================
Kflod했을때
ARDRegression 의 정답률 : [0.73467803 0.70824279 0.73102841 0.67280234 0.63055529]
AdaBoostRegressor 의 정답률 : [0.76102205 0.77947025 0.85664712 0.86888303 0.81944764]
BaggingRegressor 의 정답률 : [0.90312812 0.82989004 0.77140854 0.77993658 0.8912937 ]
BayesianRidge 의 정답률 : [0.69767107 0.7600126  0.75010456 0.57987022 0.74254604]
CCA 의 정답률 : [0.62259195 0.79808584 0.70007812 0.69064775 0.56001395]
DecisionTreeRegressor 의 정답률 : [0.67568566 0.632912   0.65748883 0.65088294 0.85592305]
DummyRegressor 의 정답률 : [-0.00349383 -0.00992912 -0.02679561 -0.00329661 -0.00299253]
ElasticNet 의 정답률 : [0.66264239 0.68334678 0.58258441 0.70485485 0.66956431]
ElasticNetCV 의 정답률 : [0.65564759 0.68792608 0.57889948 0.63298751 0.65931338]
ExtraTreeRegressor 의 정답률 : [0.35952891 0.63799857 0.76967417 0.83555717 0.56227478]
ExtraTreesRegressor 의 정답률 : [0.79166003 0.94064336 0.84969582 0.87006799 0.82714613]
GammaRegressor 의 정답률 : [-0.01159413 -0.00223821 -0.00135494 -0.0004643  -0.01346997]
GaussianProcessRegressor 의 정답률 : [-5.38913063 -7.24993982 -5.66800064 -8.97077274 -5.76040659]
GeneralizedLinearRegressor 의 정답률 : [0.70515619 0.61060441 0.69363016 0.56889932 0.57572395]
GradientBoostingRegressor 의 정답률 : [0.90481983 0.79137981 0.77519067 0.89209037 0.89244509]
HistGradientBoostingRegressor 의 정답률 : [0.75441286 0.91895497 0.84204994 0.68179434 0.87774125]
HuberRegressor 의 정답률 : [0.56067556 0.59130422 0.5185862  0.71714378 0.59905979]
IsotonicRegression 의 정답률 : [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : [0.56070617 0.41639605 0.45980596 0.49605341 0.29760067]
KernelRidge 의 정답률 : [0.47452205 0.71515463 0.76366141 0.64386103 0.57383453]
Lars 의 정답률 : [0.72997904 0.53150884 0.73528882 0.64497247 0.66383112]
LarsCV 의 정답률 : [0.61319288 0.73778093 0.69311967 0.68914988 0.76482367]
Lasso 의 정답률 : [0.63935613 0.59928436 0.64502674 0.68195099 0.67641057]
LassoCV 의 정답률 : [0.6395853  0.59891815 0.60159307 0.72099642 0.76251525]
LassoLars 의 정답률 : [-0.07712705 -0.00114647 -0.02711464 -0.00069871 -0.03760979]
LassoLarsCV 의 정답률 : [0.65530218 0.77690304 0.77195512 0.66635911 0.63032562]
LassoLarsIC 의 정답률 : [0.6464163  0.55231598 0.75030981 0.66368066 0.79810106]
LinearRegression 의 정답률 : [0.7037087  0.65031706 0.7129308  0.68518827 0.72695689]
LinearSVR 의 정답률 : [0.49297158 0.57410151 0.64037134 0.47067345 0.42646551]
MLPRegressor 의 정답률 : [0.43213636 0.6444184  0.64877201 0.55140302 0.51675395]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 : [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 : [nan nan nan nan nan]
MultiTaskLasso 의 정답률 : [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 : [nan nan nan nan nan]
NuSVR 의 정답률 : [0.0808034  0.20599101 0.22625967 0.30573869 0.17659584]
OrthogonalMatchingPursuit 의 정답률 : [0.54718886 0.43125697 0.6465036  0.5433422  0.49385711]
OrthogonalMatchingPursuitCV 의 정답률 : [0.68226713 0.70024865 0.71047451 0.618064   0.53863144]
PLSCanonical 의 정답률 : [-2.30396305 -3.34020394 -1.80365548 -1.58566471 -2.27771647]
PLSRegression 의 정답률 : [0.54961089 0.71623845 0.44006473 0.66129259 0.78070722]
PassiveAggressiveRegressor 의 정답률 : [-5.96656459 -0.59259598 -0.84370696 -0.53003285  0.33868218]
PoissonRegressor 의 정답률 : [0.81384687 0.7009566  0.73108079 0.7316365  0.70817347]
RANSACRegressor 의 정답률 : [0.74941273 0.65586645 0.40674933 0.5673629  0.43814468]
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 : [0.84303104 0.89102193 0.90185175 0.83791855 0.84048454]
RegressorChain 은 없는 놈!
Ridge 의 정답률 : [0.70710477 0.79322682 0.69436539 0.69274574 0.4672265 ]
RidgeCV 의 정답률 : [0.72426943 0.66854071 0.59052548 0.79985935 0.71249023]
SGDRegressor 의 정답률 : [-3.76278372e+26 -8.15808194e+26 -5.36467228e+25 -2.30452314e+26
 -4.14291034e+25]
SVR 의 정답률 : [0.1913773  0.2491242  0.15699578 0.06570487 0.24658734]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 : [0.72041494 0.59184565 0.6084735  0.79395048 0.56494379]
TransformedTargetRegressor 의 정답률 : [0.76538996 0.567591   0.75126532 0.73744624 0.68454983]
TweedieRegressor 의 정답률 : [0.45162774 0.50930739 0.68133692 0.72619227 0.75091608]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답률 : [nan nan nan nan nan]
'''
