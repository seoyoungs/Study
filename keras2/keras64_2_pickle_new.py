# 가중치 저장할 것
# 1. model.save() 쓸것
# 2. pickle 쓸것
# mnist 다중분류
# 원핫 인코딩에서 왜 10,2 되는지?

import numpy as np
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV

from keras.utils import np_utils
import gzip, numpy
import pickle as cPickle
import pandas as pd
import pickle, gzip, numpy, urllib.request, json

# Load the dataset
with gzip.open('C:/data/npy/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

x_train, y_train = train_set
x_test, y_test = test_set

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

print('x_train shape : ',x_train.shape)
print('x_test shape : ',x_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)

# ============== 1. 데이터/ 전처리 ============================
# x_train = x_train.reshape(50000, 28*28).astype('float32')/255.
# x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 원 핫 인코딩

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train.reshape(-1, 20)
y_test = y_test.reshape(-1, 20)

print('y_train shape : ',y_train.shape) # (50000, 10)
print('y_test shape : ',y_test.shape) # (10000, 10)


# ================ 2. 모델 ===========================
def bulid_model(drop=0.5, optimizer='adam'):
    
    inputs = Input(shape = (28*28) , name= 'input')
    x = Dense(512, activation='relu', name='Hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(20, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
    return model

# ================= 3. 모델 훈련 ==========================
def create_hyperparameters():
    batches = [40, 50] # 이거 늘리기
    optimizers = ['adam'] #'rmsprop','adam','adadelta'
    dropout = [0.2,0.25]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}

hyperparameters  = create_hyperparameters()
model2 = bulid_model()
model2.save('../data/h5/k64_1_model.h5')

# 데이터 불러오기
# model3 = load_model('../data/h5/k64_model.h5')

# sklearn에서data 인식하게 rap으로 감싸기
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# KerasClassifier 여기도 epochs 가능하다
model2 = KerasClassifier(build_fn=bulid_model, verbose=1, epochs=30,
                         validation_split=0.2)

# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함
search = RandomizedSearchCV(model2, hyperparameters , cv=2)

search.fit(x_train,y_train, verbose=1) # # fit에 callback 함수도 가능하다

print("#################################")
print(search.best_params_)
print(search.best_score_)
print("#################################")
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)


