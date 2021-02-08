# 61번을 파이프라인으로 구성
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
params = create_hyperparameters()
model2 = bulid_model()

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

model2 = make_pipeline(MinMaxScaler(), model2)


# sklearn에서data 인식하게 rap으로 감싸기
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# KerasClassifier 여기도 epochs 가능하다
model2 = KerasClassifier(build_fn=bulid_model, verbose=1, epochs=50,
                         validation_split=0.2)
# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함
search = RandomizedSearchCV(model2, params, cv=3)

# fit에 callback 함수도 가능하다
def callbacks():
    modelpath = '../data/modelCheckpoint/k65_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
    es = EarlyStopping(monitor = 'val_loss',patience=5)
    cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
    return es,lr,cp

es,lr,cp = callbacks()

search.fit(x_train,y_train,verbose=1, epochs=100, 
                validation_split=0.2, callbacks = [es, cp, lr])

# search.save('../data/h5/k65_1_model.h5')

acc = search.score(x_test,y_test)
print("최종스코어 : ",acc)
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)

