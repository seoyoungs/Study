from tensorflow.keras.applications import VGG16

model = VGG16(weights = 'imagenet', include_top=False, input_shape=(32, 32, 3))
model = VGG16() # 이렇게 default로 훈련해도 된다
model.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
model.summary()
print(len(model.weights)) # 26
print(len(model.trainable_weights)) # 0

# default인 것 아닌것, 시간차이가 많이 난다
'''
vgg16(include_top=False)
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
26
0

디폴트 일 때 => vgg16(include_top=True)
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
32
0
'''
