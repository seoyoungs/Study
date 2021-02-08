# epochs =100
# validation_split, callback 적용
# earlystopping 5 
# reduce LR 3
# 필수 : 노드의 갯수
# modelcheckpoint 폴더에 hdf5저장(val_loss포함)

import numpy as np
from tensorflow.keras.models import Sequential,Model
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

x_train = x_train.reshape(-1,28*28).astype('float32')/255.0
x_test = x_test.reshape(-1,28*28).astype('float32')/255.0

x_train= x_train.reshape(-1,28*28,1)
x_test= x_test.reshape(-1,28*28,1)

# ================ 2. 모델 ===========================
def build_model(drop=0.5,optimizer='adam'):
    model = Sequential() #모델 생성.
    model.add(Conv1D(128, kernel_size=1, padding='same', strides=1, 
      activation='relu', input_shape=(784, 1)))  #Conv1D 모델을 추가한다.
    model.add(MaxPooling1D(2)) #Maxpooling 추가를 한다.
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64,3, activation='softmax'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax')) 
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics = ['acc'])
    return model

# ================= 3. 모델 훈련 ==========================
def create_hyperparameters():
    batches = [20, 40, 50] # 이거 늘리기
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.2,0.25,0.3]
    return {'batch_size':batches,'optimizer':optimizers,'drop':dropout}

# sklearn에서data 인식하게 rap으로 감싸기
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# KerasClassifier 여기도 epochs 가능하다
model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=50,
                         validation_split=0.2)
params = create_hyperparameters()

# model2 = build_model() ## 이렇게 하면 에러난다 sklearn에서 keras모델 인식을 못함
search = RandomizedSearchCV(model2, params, cv=3)

# fit에 callback 함수도 가능하다
modelpath = '../data/modelCheckpoint/k61_4_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                    save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, mode='auto')
search.fit(x_train,y_train,verbose=1, epochs=100, 
                validation_split=0.2, callbacks = [es, cp, lr])

acc = search.score(x_test,y_test)
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
print("최종스코어 : ",acc)

'''
{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 50}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000027B07640460>  
0.9537333448727926
최종스코어 :  0.9635000228881836
'''

