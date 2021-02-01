### randomforest로 모델링 --> PCA로 압축

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

# 전처리==================================
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state=44
)


pca = PCA(n_components=7) # 파라미터 주성분 개수
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

cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
 0.94794364 0.99131196 0.99914395 1.        ]

이렇게 보면 n_components를 어디서 잡아야하는지 알 수 있다

d= np.argmax(cumsum >= 0.95)+1 # 결과값에서 가장 큰 값 추출
print('cumsum  >= 0.95', cumsum >= 0.95)
print('d :', d) # 머신러닝은 차원축소 기술이 좋다

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
'''

model = XGBRegressor(n_jobs=-1, use_label_encoder=False)

# 훈련
model.fit(x_train,y_train, eval_metric='logloss')

#4. 평가, 예측
acc = model.score(x_test, y_test) #model.evaluate 와 같음

# print(model.feature_importances_) #feature가 많다고 좋은 것아님(곡선화, 과적합 될 수 있음)
print('acc: ', acc)

'''
DecisionTreeClassifier
acc:  0.22679038813478902
acc :  0.3284929255077368

RandomForestRegressor
acc:  0.4035535261514329
acc :  0.5204110010105024

GradientBoostingRegressor
acc:  0.3675141238137092
acc :  0.4178561105408377

XGBRegressor
acc:  0.24138193114785134
acc :  0.30441612593454836

PCA(n_components=7) , RandomForestRegressor
acc:  0.39760407502056017

PCA(n_components=7), XGBRegressor
acc:  0.24138193114785134
'''
