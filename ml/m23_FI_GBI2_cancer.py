# 실습
# ## feature중 25% 미만 제거
# GradientBoostingClassifier모델로 돌려서 acc확인

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

#2. 모델
# model = DecisionTreeClassifier(max_depth = 4)
model = GradientBoostingClassifier() # 데이터 골고루 분포된다

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

### 하위 25% 칼럼 제거하기
# # print(model.feature_importances_) 
# df = pd.DataFrame(dataset.data, )
# original = model.feature_importances_
# data_new =[]  # 새로운 데이터형성 dataset --> data_new
# feature_names = []  # 컬럼 특징 정의 feature_names
# a = np.percentile(model.feature_importances_, q=25) # percentile(백분위)로 미리 정의

# # for문 생성-> 0제거하는 것
# print(model.feature_importances_) 
df = pd.DataFrame(dataset.data, )
original = model.feature_importances_
data_new =[]  # 새로운 데이터형성 dataset --> data_new
feature_names = []  # 컬럼 특징 정의 feature_names
a = np.percentile(model.feature_importances_, q=25) # percentile(백분위)로 미리 정의

# for문 생성-> 0제거하는 것
for i in range(len(original)):
    if original[i] > a: # 하위 25% 값보다 큰 것만 추출
        data_new.append(dataset.data[:,i])
        feature_names.append(dataset.feature_names[i])

data_new = np.array(data_new)
data_new = np.transpose(data_new)
x_train,x_test,y_train,y_test = train_test_split(data_new,dataset.target,
                                                 train_size = 0.8, random_state = 33)

#2. 모델
model = GradientBoostingClassifier(max_depth = 4)

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
DecisionTreeClassifier으로 할 때
[0.         0.         0.         0.         0.         0.
 0.         0.01297636 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.0629016  0.78678449 0.
 0.00995429 0.01008994 0.         0.11729332 0.         0.        ]
acc:  0.9385964912280702
[0.06144428 0.06551738 0.76116335 0.0137178  0.0110841  0.08707309]
acc :  0.9298245614035088         # 칼럼 지운 후


RandomForestClassifier 으로 할때
[0.02765248 0.01485853 0.03946768 0.04081673 0.00574651 0.01467651
 0.05058996 0.09139004 0.00450387 0.00375983 0.01450889 0.00383512
 0.0169872  0.05446694 0.00430911 0.00571383 0.00856116 0.00634666
 0.00341841 0.00531692 0.10639541 0.02521005 0.15013684 0.10582912
 0.01139022 0.01410134 0.0215164  0.12489599 0.01138414 0.0122141 ]
acc:  0.9649122807017544
[0.06336356 0.01619906 0.03508036 0.04117757 0.00637716 0.08212835
 0.07385174 0.00890802 0.00237418 0.02578177 0.0046971  0.00643837
 0.1213598  0.01176054 0.19143813 0.14267937 0.01059104 0.01570933
 0.03387219 0.09147605 0.0083154  0.0064209 ]
acc :  0.9298245614035088                  # 25% 자른 후

GradientBoostingClassifier으로 할 때
acc:  0.9824561403508771
acc :  0.9035087719298246
'''


