# 가중치 저장할 것
# 1. model.save() 쓸것
# 2. pickle 쓸것

import numpy as np
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV
import pickle

(x_train,y_train),(x_test,y_test) = mnist.load_data()


# ============== 1. 데이터/ 전처리 ============================
# 원 핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# ================ 2. 모델 ===========================
def bulid_model(drop=0.5, optimizer='adam'):
    
    inputs = Input(shape=(28*28,), name='Input')
    x = Dense(512, activation='relu', name='Hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
    return model

# ================= 3. 모델 훈련 ==========================
def create_hyperparameters():
    batches = [20, 40, 50] # 이거 늘리기
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.2,0.25,0.3]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}

hyperparameters  = create_hyperparameters()
model2 = bulid_model()
model2.save('../data/h5/k64_1_model.h5')

# 데이터 불러오기
# model3 = load_model('../data/h5/k64_model.h5')

# sklearn에서data 인식하게 rap으로 감싸기
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# KerasClassifier 여기도 epochs 가능하다
model2 = KerasClassifier(build_fn=bulid_model, verbose=1, epochs=50,
                         validation_split=0.2)
import pickle
pickle.dump(model2, open('../data/npy/keras64.pickle.dat', 'wb'))
# 피클 위치!!!!!

model4 = pickle.load(open('../data/npy/keras64.pickle.dat', 'rb'))
# 피클 로드

# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함
search = RandomizedSearchCV(model2, hyperparameters , cv=3)

search.fit(x_train,y_train, verbose=1) # # fit에 callback 함수도 가능하다

print("#################################")
print(search.best_params_)
print(search.best_score_)
print("#################################")
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)

'''
{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 20}
0.975350002447764
최종 스코어 :  0.9797000288963318
'''
