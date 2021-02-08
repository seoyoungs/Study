# 분류

import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

dataset = load_breast_cancer()
x= dataset.data
y= dataset.target

# ============== 1. 데이터/ 전처리 ============================

x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

print(x_test.shape, x_train.shape, y_test.shape, y_train.shape) 
#(114, 30) (455, 30) (114,) (455,)

scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# ================ 2. 모델 ===========================
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = 30 , name= 'input')
    x = Dense(128, activation='relu', name ='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name ='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu', name ='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='sigmoid', name ='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

# ================= 3. 모델 훈련 ========================
def create_hyperparameters():
    batches = [10,20,30,40,50] # 이거 늘리기
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.1,0.2,0.3]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}
# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함

# keras의 mnist data를 sklearn에서 인식하게 wrappers로 감싸기
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
search.fit(x_train,y_train,verbose=1)

acc = search.score(x_test,y_test)
print('최적의 매개변수: ', search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
print("최종스코어 : ",acc)
# y_pred = search.predict(x_test)
# print('최종정답률', accuracy_score(y_test, y_pred))
'''
최적의 매개변수:  {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001C19F754610>
0.8681015372276306
최종스코어 :  0.8508771657943726
'''


