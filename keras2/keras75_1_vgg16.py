from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights = 'imagenet', include_top=False, input_shape=(32, 32, 3))
# print(model.weights)

# model.trainable = False # 훈련을 안시키겠다, 저장된 가중치 사용
# model.summary()
# # 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
# print(len(model.weights)) # 26
# print(len(model.trainable_weights)) # 0
# # bias 성향
'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
'''

model.trainable = True # 훈련을 시키겠다
model.summary()
# 즉, 16개의 레이어지만 연산되는 것은 13개 이고 그래서 len=26개
print(len(model.weights)) # 26
print(len(model.trainable_weights)) # 26
'''
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''