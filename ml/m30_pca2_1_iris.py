### randomforest로 모델링 --> PCA로 압축

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (150, 4) (150,)

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state=44
)

pca = PCA(n_components=3) # 파라미터 주성분 개수
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


cumsum : [0.92461872 0.97768521 0.99478782 1.        ]

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
[0.00787229 0.         0.4305627  0.56156501]
acc:  0.9333333333333333
[0.02899179 0.0539027  0.91710551]
acc :  0.8666666666666667             # 칼럼 지운 후 

RandomForestClassifier
[0.00787229 0.         0.4305627  0.56156501]
acc:  0.9333333333333333
[0.02899179 0.0539027  0.91710551]
acc :  0.8666666666666667             # 칼럼 지운 후 

GradientBoostingClassifier
[0.00542723 0.01237517 0.62262084 0.35957677]
acc:  0.9666666666666667
[0.16023808 0.37594413 0.46381779]
acc :  0.9

XGBClassifier
acc :  0.9666666666666667

PCA(n_components=3), RandomForestClassifier
acc:  0.9666666666666667

pca = PCA(n_components=3), XGBClassifier
acc:  0.9666666666666667
'''

