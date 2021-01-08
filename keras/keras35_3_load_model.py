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
model = load_model("./model/save_keras35.h5")
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
'''