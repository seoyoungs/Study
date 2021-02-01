# 실습
# ## feature중 25% 미만 제거
# GradientBoostingClassifier모델로 돌려서 acc확인

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier #Gradient 기반이나 더 빠르다, 과적합 방지
import pandas as pd
import timeit# 시간측정
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

#1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state=44
)

# 타임 걸기 (-1,1,4,8비교)
#--------------------------------------------------------------------------------
start_time = timeit.default_timer() # 시작 시간 체크

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

####=============시간걸기====================================
terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time))
#============================================================
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
DecisionTreeClassifier으로 할 때
acc:  0.9385964912280702
acc :  0.9298245614035088         # 칼럼 지운 후


RandomForestClassifier 으로 할때
acc:  0.9649122807017544
acc :  0.9298245614035088                  # 25% 자른 후

GradientBoostingClassifier으로 할 때
acc:  0.9824561403508771
acc :  0.9035087719298246

XGBClassifier
acc:  0.9824561403508771
acc :  0.9473684210526315

시간비교 
(n_jobs=-1) 0.069513초 걸렸습니다.
(n_jobs=1) 0.113210초 걸렸습니다.
(n_jobs=4) 0.075029초 걸렸습니다.
(n_jobs=8) 0.061862초 걸렸습니다.
---> 즉, 병렬인데 8이 더 빠르다(즉, 빠르기 큰 효과는 없다)
'''

