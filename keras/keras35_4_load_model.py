#35_3카피
import numpy as np
a= np.array(range(1,11))
size=5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] #위치 잘 맞춰서 넣어주기
        aaa.append(subset)
        #aaa.append([item for item  in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
#print('------------------')
#print(dataset)

x = dataset[:,0:4]
y = dataset[:,-1] 

x=x.reshape(6,4,1)

from tensorflow.keras.models import load_model
model = load_model("../data/h5/test_save_keras35_1.h5")

#요 밑 3줄 넣고 테스트######
from tensorflow.keras.layers import Dense
model.add(Dense(5, name='kingkeras1'))  #레이어의 이름 dense
model.add(Dense(1, name='kingkeras2')) #레이어의 이름 dense_1
############### 이렇게 이름 충돌안나게 지정하면 기존 레이어 밑으로 들어간다.
'''
ValueError: All layers added to a Sequential model should 
have unique names. Name "dense" is already the name of 
a layer in this model. Update the `name` argument to pass a unique name.
그럼 이렇게 에러난다. 왜? 기존 저장된 dense와 충돌한다.
'''
model.summary()

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=10, batch_size=1, verbose=1)

#4. 평가, 예측
loss= model.evaluate(x,y)
print('loss :', loss)
y_predict=model.predict(x)
print(y_predict)

'''
이 에러 왜 뜨는지 알아보기(가중치 할 때 알아볼 것)
WARNING:tensorflow:No training configuration found in the save file, 
so the model was *not* compiled. Compile it manually.
결과
loss : 0.10717002302408218
[[ 4.777073 ]
 [ 6.202416 ]
 [ 7.441399 ]
 [ 8.485246 ]
 [ 9.3462515]
 [10.046528 ]]
 레이어 추가후
loss : 0.1975240707397461
[[4.9222093]
 [6.4865255]
 [7.687375 ]
 [8.577417 ]
 [9.231223 ]
 [9.711857 ]]
'''