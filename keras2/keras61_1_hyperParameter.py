import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ============== 1. 데이터/ 전처리 ============================
from tensorflow.keras.utils import to_categorical
 
# 원 핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28*28).astype('float32')/255.0
x_test = x_test.reshape(-1,28*28).astype('float32')/255.0

# ================ 2. 모델 ===========================
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape = (28*28) , name= 'input')
    x = Dense(512, activation='relu', name ='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name ='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name ='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name ='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer, metrics = ['acc'])
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
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
print("최종스코어 : ",acc)

'''
{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000286C95B5EB0>
0.9574833114941915
최종스코어 :  0.9581000208854675
'''
