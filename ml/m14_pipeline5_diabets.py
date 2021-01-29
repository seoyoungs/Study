import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝에 대해서 무시

dataset = load_diabetes()
x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

# 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델링  
# model = Pipeline([('scaler', MinMaxScaler()), 
#              ('maliddong', SVC())]) # MinMaxScaler와 SVC 합치기, ''안에 이름은 자유
#전처리 한개와 모델 한개 합친것
model = make_pipeline(StandardScaler(), RandomForestRegressor()) #이름 지정없이 쓸수있다

# 훈련
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

'''
MinMaxScaler, RandomForestRegressor
0.4967677523428534

StandardScaler
0.511461526415514
'''