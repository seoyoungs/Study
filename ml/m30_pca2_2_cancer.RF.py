### randomforest로 모델링 --> PCA로 압축

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) # (569, 30) (569,)

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state=44
)

pca = PCA(n_components=8) # 파라미터 주성분 개수
x2 = pca.fit_transform(x) #np.transformㅘ 같음
# print(x2.shape) #(442, 8) 이렇게 컬럼 재구성

pca_EVR = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율
print(pca_EVR) # 8개로 줄인 중요도에 대한 수치
print(sum(pca_EVR)) #0.9913119559917795 (8개로 줄였을 때의 압축률)
'''
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum :', cumsum) # 컬럼10개라 10번 돌아간다


cumsum : [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.        ]

이렇게 보면 n_components를 어디서 잡아야하는지 알 수 있다
(누적 비율 계산)

d= np.argmax(cumsum >= 0.95)+1 # 결과값에서 가장 큰 값 추출
print('cumsum  >= 0.95', cumsum >= 0.95) #0.95는 내가 임의로 설정
print('d :', d) # 머신러닝은 차원축소 기술이 좋다

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
'''

model = XGBClassifier(n_jobs=-1, use_label_encoder=False)

# 훈련
model.fit(x_train,y_train, eval_metric='logloss')

#4. 평가, 예측
acc = model.score(x_test, y_test) #model.evaluate 와 같음

# print(model.feature_importances_) #feature가 많다고 좋은 것아님(곡선화, 과적합 될 수 있음)
print('acc: ', acc)

'''
DecisionTreeClassifier으로 할 때
acc:  0.9385964912280702
acc :  0.9298245614035088         # 칼럼 지운 후


RandomForestClassifier 으로 할때
acc:  0.9649122807017544
acc :  0.9298245614035088                  # 25% 자른 후

GradientBoostingClassifier으로 할 때
acc:  0.9824561403508771
acc :  0.9035087719298246

XGBClassifier
acc:  0.9824561403508771
acc :  0.9473684210526315

시간비교 
(n_jobs=-1) 0.069513초 걸렸습니다.
(n_jobs=1) 0.113210초 걸렸습니다.
(n_jobs=4) 0.075029초 걸렸습니다.
(n_jobs=8) 0.061862초 걸렸습니다.
---> 즉, 병렬인데 8이 더 빠르다(즉, 빠르기 큰 효과는 없다)

PCA(n_components=8), RandomForestClassifier
acc:  0.9649122807017544

pca = PCA(n_components=8), XGBClassifier
acc:  0.9824561403508771
'''
