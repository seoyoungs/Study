## pca 컬럼개수 줄인다(중요컬럼 대상으로 축소)
# FI 중요한 컬럼 확인만

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=8) # 파라미터 주성분 개수
x2 = pca.fit_transform(x) #np.transformㅘ 같음
print(x2.shape) #(442, 7) 이렇게 컬럼 재구성

pca_EVR = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율
print(pca_EVR) # 7개로 줄인 중요도에 대한 수치
print(sum(pca_EVR)) #0.9479436357350411 (7개로 줄였을 때의 압축률)

# sum(pca_EVR)  7: 0.9479436357350411
# 8 : 0.9913119559917795
# 통상적으로 95% 정도 되면 모델 성능 거의 포함시킨다고 보면 됨

