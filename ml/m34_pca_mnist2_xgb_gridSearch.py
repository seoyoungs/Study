# m31로 만든 0.95 이상의 n_component =?를 사용해
# dnn 모델 만들어라
# mnist dnn 보다 성능 좋게
## 이때 전처리시 y는 따로 부여하지 않는다---> 그럼 훈련, 평가는 어떻게??

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape) # (70000, 28, 28)
print(y.shape) # (70000,)

x = x.reshape(70000, 28*28)
print(x.shape) # (70000, 784)

#======================== pca ===============================
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
# print("cumsum : ", cumsum) 

d = np.argmax(cumsum > 1.0)+1
# print("cumsum >= 0.95", cumsum >= 0.95) 
print("d : ", d) # d :  154

pca = PCA(n_components=d)
x = pca.fit_transform(x)
# print(x)
print(x.shape) # (70000, 154)


#===============================다중분류 y원핫코딩& 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

#부분만 전처리 해주기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)
'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  
# print(x_train.shape, x_test.shape) # (56000, 154) (14000, 154)
# print(y_train.shape, y_test.shape) # (56000,) (14000,)
'''

##파라미터 튜닝 & 모델링 =============================================

kflod = KFold(n_splits=5, shuffle=True)

parameter = [
    {'n_estimators' : [100,200]},
    {'max_depth': [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1]} #총15번
]

model = GridSearchCV(XGBClassifier(n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False), parameter, cv=kflod)
# score = cross_val_score(model, x2_train, y2_train, cv= kflod) #GridSearchCV에서 나온값, 또 5번해서 최적의 값
# 그럼 총 25번
# print(score.shape)
# print('교차검증점수 : ', score)

# ===================================== 3. 훈련
model.fit(x_train,y_train, eval_metric='logloss', verbose=True) # eval_set=[(x_train, x_test), (y_train, y_test)])
# eval_metric='error'는 accuracy와 같다, 
# eval_set=[(x_train, x_test), (y_train, y_test)] -> validation_data와 같은기능

#========================================== 4. 평가
acc = model.score(x_test, y_test)
print('acc: ', acc)


'''
Mnist
결과
[0.11182762682437897, 0.9659000039100647]

Dnn
loss: [0.18929392099380493, 0.9452999830245972]
y_pred:  [7 2 1 0 4 1 4 9 6 9]
y_test:  [7 2 1 0 4 1 4 9 5 9]

PCA, DNN (0.95)
[0.09999978542327881, 0.899996280670166]
y_pred:  [0 0 0 0 0 0 0 0 0 0]
y_test:  [8 4 6 2 6 1 0 7 9 7]

PCA, DNN (1.0)
[0.09999978542327881, 0.899996280670166]

PCA, XGB (0.95)
acc :  0.9647142857142857

PCA, XGB (1.0)
acc :  0.9647142857142857

'''

