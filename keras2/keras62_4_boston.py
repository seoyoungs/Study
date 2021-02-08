
# 회귀
# acc는 회귀에서 0 , mae로 하기
# KerasRegressor

import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import warnings
warnings.filterwarnings("ignore")

dataset = load_boston()
x= dataset.data
y= dataset.target

# ============== 1. 데이터/ 전처리 ============================

x_train, x_test, y_train, y_test= train_test_split(x,y, 
                     shuffle=True, train_size=0.8, random_state=66)

print(x_test.shape, x_train.shape, y_test.shape, y_train.shape) 
#(102, 13) (404, 13) (102,) (404,)

scaler =MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# ================ 2. 모델 ===========================
def build_model(drop=0.5,optimizer='adam'):
    inputs = Input(shape = 13 , name= 'input')
    x = Dense(128, activation='relu', name ='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name ='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu', name ='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='relu', name ='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ================= 3. 모델 훈련 ========================
def create_hyperparameters():
    batches = [30,50,60,40,80] # 이거 늘리기
    optimizers = ['rmsprop','adam','adadelta', 'sgd']
    dropout = [0.2,0.25,0.3]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}
# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함

# keras의 mnist data를 sklearn에서 인식하게 wrappers로 감싸기
model2 = KerasRegressor(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=2)
search.fit(x_train,y_train, epochs=10, verbose=1)

mae = search.score(x_test,y_test)
print('최적의 매개변수: ', search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
loss = search.score(x_test, y_test)
print("최종스코어 : ",loss)

'''
최적의 매개변수:  {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 30}
<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001B500B09A00>
-93.7008285522461
4/4 [==============================] - 0s 1ms/step - loss: 40.7555 - mae: 4.4575
최종스코어 :  -40.75551986694336
'''

