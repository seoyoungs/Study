# m31로 만든 1.0 이상의 n_component =?를 사용해
# dnn 모델 만들어라
# mnist dnn 보다 성능 좋게
## 이때 전처리시 y는 따로 부여하지 않는다---> 그럼 훈련, 평가는 어떻게??

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape) # (70000, 28, 28)
print(y.shape) # (70000,)

x = x.reshape(70000, 28*28)
print(x.shape) # (70000, 784)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
# print("cumsum : ", cumsum) 

d = np.argmax(cumsum > 0.95)+1
# print("cumsum >= 0.95", cumsum >= 0.95) 
print("d : ", d) # d :  154

pca = PCA(n_components=d)
x = pca.fit_transform(x)
# print(x)
print(x.shape) # (70000, 154)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
print(x_train.shape, x_test.shape) # (56000, 154) (14000, 154)
print(y_train.shape, y_test.shape) # (56000,) (14000,)

kfold = KFold(n_splits=5, shuffle=True)

model = XGBClassifier(n_jobs=8, use_label_encoder=False)

# 3. Train
model.fit(x_train, y_train, eval_metric='logloss')

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)



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