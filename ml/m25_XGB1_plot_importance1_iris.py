# 실습
# ## feature중 25% 미만 제거
# GradientBoostingClassifier모델로 돌려서 acc확인

from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cut_columns(feature_importances,columns,number):
    temp = []
    # print(len(feature_importances))
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

# 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

# 모델
model = XGBClassifier(n_jobs=-1, use_label_encoder=False)

# 훈련
model.fit(x_train,y_train, eval_metric='logloss')

# 평가, 예측
acc = model.score(x_test,y_test)
print(model.feature_importances_)
# print(dataset.feature_names)
print("acc : ",acc)
'''
def plot_feature_importances_datasets(model,datasets):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_datasets(model,dataset)
'''

plot_importance(model) #XGB에서 위 복잡한 importances식 제공(f_score)
plt.show()


# df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
# df.drop(cut_columns(model.feature_importances_,dataset.feature_names,2),axis=1,inplace=True)
# print(cut_columns(model.feature_importances_,dataset.feature_names,2))

# x2_train, x2_test, y2_train, y2_test = train_test_split(
#     dataset.data, dataset.target, train_size = 0.8, random_state=44
# )
# model = XGBClassifier(n_jobs=-1)

# # 훈련
# model.fit(x2_train,y2_train)
# y = dataset.target

# # 평가, 예측
# acc = model.score(x2_test,y2_test)
# print("acc : ",acc)

'''
DecisionTreeClassifier
[0.00787229 0.         0.4305627  0.56156501]
acc:  0.9333333333333333
[0.02899179 0.0539027  0.91710551]
acc :  0.8666666666666667             # 칼럼 지운 후 

RandomForestClassifier
[0.00787229 0.         0.4305627  0.56156501]
acc:  0.9333333333333333
[0.02899179 0.0539027  0.91710551]
acc :  0.8666666666666667             # 칼럼 지운 후 

GradientBoostingClassifier
[0.00542723 0.01237517 0.62262084 0.35957677]
acc:  0.9666666666666667
[0.16023808 0.37594413 0.46381779]
acc :  0.9

XGBClassifier
acc :  0.9666666666666667
'''