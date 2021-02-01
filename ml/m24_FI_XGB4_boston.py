from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import timeit# 시간측정
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

#1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

#2. 모델
# model = DecisionTreeClassifier(max_depth = 4)
model = XGBClassifier(n_jobs=-1, use_label_encoder=False)

# 훈련
model.fit(x_train,y_train, eval_metric='logloss')

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
model = XGBClassifier(n_jobs=-1, use_label_encoder=False)

# 훈련
model.fit(x_train,y_train, eval_metric='logloss')

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
'''

