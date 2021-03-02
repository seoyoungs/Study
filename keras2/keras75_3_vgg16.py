from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(32, 32, 3))
# print(model.weights)

vgg16.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
vgg16.summary()
# 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
print(len(vgg16.weights)) # 26
print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16) # 3차원 -> layer 26개
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
'''
Total params: 14,719,879
Trainable params: 5,191     # model부분만 훈련
Non-trainable params: 14,714,688
'''
print('그냥 가중치의 수 :',len(model.weights)) # 32 -> ((13+3)*2)
print('동결하기 전 훈련되는 가중치의 수', len(model.trainable_weights)) # 6 :model부분 연산가능 레이어 3개 *2

'''
len(model.trainable_weights)
가져온 모델을 훈련 안시키려고 동결시켜놓은 다음에 내가 훈련할 가중치의 수
'''
######################### 파일 분리

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns= ['Layer Type', 'Layer name', 'Layer Trainable']) #layers이름 지정

print(aaa)


