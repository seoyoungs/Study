from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

#2. 모델
model = DecisionTreeClassifier(max_depth = 4)

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

'''
[0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.00677572 0.         0.         0.         0.
 0.         0.         0.01008994 0.05612587 0.78000877 0.
 0.02293065 0.         0.         0.11729332 0.         0.00677572]
acc:  0.9385964912280702
'''


