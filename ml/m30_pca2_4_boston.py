### randomforest로 모델링 --> PCA로 압축

import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

# 전처리==================================
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state=44
)


pca = PCA(n_components=5) # 파라미터 주성분 개수
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

cumsum : [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
 0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
 1.        ]

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
acc:  0.8900975502183226
acc :  0.8048681892684204

RandomForestRegressor
acc:  0.8159350178696477
acc :  0.645434902587047

GradientBoostingRegressor
acc:  0.8951681035441386
acc :  0.85420509829295

XGBRegressor
acc:  0.8902902185916939
acc :  0.8504126332900991

PCA(n_components=5), RandomForestRegressor
acc:  0.8925715106072507

PCA(n_components=5), XGBRegressor
acc:  0.8902902185916939
'''
