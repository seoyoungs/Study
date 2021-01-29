import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_iris()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

# 전처리
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 파라미터 -> 이전에 kfold넣어도 됨
# parameters = [
#     {"svc__C" : [1, 10, 100, 1000], 'svc__kernel':['linear']},#4번
#     {"svc__C" : [1, 10, 100], 'svc__kernel':['ref'], 'svc__gamma':[0.001, 0.0001]},#6번
#     {"svc__C" : [1, 10, 100, 1000], 'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}#8번
# ] #총 18번을 n_splits=5로 18*5 =90번 돈다(이것은 우리가 정하는 것)
# # svc__ 언더바가 2개 들어간다 --> 문법임

parameters = [
    {"mal__C" : [1, 10, 100, 1000], 'mal__kernel':['linear']},#4번
    {"mal__C" : [1, 10, 100], 'mal__kernel':['ref'], 'mal__gamma':[0.001, 0.0001]},#6번
    {"mal__C" : [1, 10, 100, 1000], 'mal__kernel':['sigmoid'], 'mal__gamma':[0.001, 0.0001]}#8번
]

# 모델링  
pipe = Pipeline([('scaler', MinMaxScaler()), #전처리를 train만 한다.(훈련할때마다 달라짐)
              ('mal', SVC())]) # MinMaxScaler와 SVC 합치기, ''안에 이름은 자유
#전처리 한개와 모델 한개 합친것
# pipe = make_pipeline(StandardScaler(), SVC()) #이름 지정없이 쓸수있다

# model = GridSearchCV(pipe, parameters, cv=5) #---> 꼭 위Pipeline 형태로 써야 함
model = RandomizedSearchCV(pipe, parameters, cv=5) #---> 꼭 위Pipeline 형태로 써야 함
# 여기 cv 여러번 쓸 수 있다

# 훈련
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

'''
svc__(언더바 두개), mal
0.8666666666666667
'''

