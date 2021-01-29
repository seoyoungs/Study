# 실습
# ## feature중 0인것 없애기 
# DecisionTree모델로 돌려서 acc확인

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

#2. 모델
model = DecisionTreeRegressor(max_depth = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test) #model.evaluate 와 같음

print(model.feature_importances_) #feature가 많다고 좋은 것아님(곡선화, 과적합 될 수 있음)
print('acc: ', acc)
# feature = x_train, x_test, y_train, y_test(max_depth = 4)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1] # y
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center') # barh는 가로 막대 그래프
    plt.yticks(np.arange(n_features), dataset.feature_names) # y축 세부설정
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

## 0인 컬럼 제거
# print(model.feature_importances_) 
original = model.feature_importances_
data_new =[]  # 새로운 데이터형성 dataset --> data_new
feature_names = []  # 컬럼 특징 정의 feature_names

# for문 생성-> 0제거하는 것
for i in range(len(original)):
    if original[i] !=0: # x != y	x와 y가 같지 않다, 즉, i와 0과 같지 않다
        data_new.append(dataset.data[:,i])
        feature_names.append(dataset.feature_names[i])

data_new = np.array(data_new)
data_new = np.transpose(data_new)
x_train,x_test,y_train,y_test = train_test_split(data_new,dataset.target,
                                                 train_size = 0.8, random_state = 33)

#2. 모델
model = DecisionTreeRegressor(max_depth = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ', acc)

####### dataset -> new_data 로 변경, feature_name 부분을 feature 리스트로 변경
def plot_feature_importances_dataset(model):
    n_features = data_new.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''

[0.05832757 0.00288921 0.         0.         0.00585666 0.61396724
 0.00580415 0.094345   0.         0.         0.01858958 0.00343964
 0.19678095]
acc:  0.8159350178696477
[4.03271450e-02 0.00000000e+00 6.97375363e-03 6.37871242e-01
 1.75822478e-02 9.18117628e-02 4.93083766e-03 1.24666268e-04
 2.00378345e-01]
acc :  0.6454349025870473

'''
