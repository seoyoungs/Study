# 컬럼 개수가 784개인 mnist를 압축시킨다
## PCA로 압축시켜 0.95 이상으로 만든다
## 이때 전처리시 y는 따로 부여하지 않는다---> 그럼 훈련, 평가는 어떻게??

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor

(x_train, _), (x_test, _) = mnist.load_data() # '_' -->y_train 과 y_test안하겠다는 것

x = np.append(x_train, x_test, axis = 0)
print(x.shape) #(70000, 28, 28)

#============================================== PCA로 컬럼 압축 
x = x.reshape(-1, 784)
# x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784) # 2차원으로 pca 하기위해

pca = PCA(n_components=155) # 파라미터 주성분 개수 
x2 = pca.fit_transform(x) #np.transform과 같음
# print(x2.shape) #(442, 8) 이렇게 컬럼 재구성

pca_EVR = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율
# print(pca_EVR) # 8개로 줄인 중요도에 대한 수치
print(sum(pca_EVR)) # 0.9504055742217271 pca에 대한 신뢰도


