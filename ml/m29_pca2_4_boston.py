import numpy as np
from sklearn.datasets import load_diabetes, load_wine, load_boston
from sklearn.decomposition import PCA

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (178, 13) (178,)
'''
pca = PCA(n_components=8) # 파라미터 주성분 개수
x2 = pca.fit_transform(x) #np.transformㅘ 같음
print(x2.shape) #(442, 8) 이렇게 컬럼 재구성

pca_EVR = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율
print(pca_EVR) # 8개로 줄인 중요도에 대한 수치
print(sum(pca_EVR)) #0.9913119559917795 (8개로 줄였을 때의 압축률)
'''

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum :', cumsum) # 컬럼10개라 10번 돌아간다
'''
cumsum : [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
 0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
 1.        ]
이렇게 보면 n_components를 어디서 잡아야하는지 알 수 있다
(누적 비율 계산)
'''
d= np.argmax(cumsum >= 0.95)+1 # 결과값에서 가장 큰 값 추출
print('cumsum  >= 0.95', cumsum >= 0.95)
print('d :', d) # 머신러닝은 차원축소 기술이 좋다(d: 1차원)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
