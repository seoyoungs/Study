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

pca = PCA(n_components=5) # 파라미터 주성분 개수
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

cumsum : [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
 0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
 1.        ]

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
DecisionTreeClassifier
acc:  0.8888888888888888
acc :  0.75  

RandomForestClassifier
acc:  0.9722222222222222
acc :  1.0

GradientBoostingClassifier
acc:  0.9166666666666666
acc :  0.75

XGBClassifier
acc:  0.9444444444444444
acc :  0.9722222222222222

pca = PCA(n_components=5), RandomForestClassifier
acc:  0.9649122807017544

pca = PCA(n_components=5), XGBClassifier
acc:  0.9824561403508771
'''
