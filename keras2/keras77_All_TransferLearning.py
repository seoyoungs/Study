from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

model = ResNet101V2()

model.trainable = True # 훈련을 시키겠다
model.summary()
print(len(model.weights)) 
print(len(model.trainable_weights))

# 모델 별로 파라미터와 웨이트 수 

'''
VGG16
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
32
32

VGG19
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
38
38

Xception
Total params: 22,910,480
Trainable params: 22,855,952
Non-trainable params: 54,528
__________________________________________________________________________________________________
236
156

ResNet101
Total params: 44,707,176
Trainable params: 44,601,832
Non-trainable params: 105,344
__________________________________________________________________________________________________
626
418
'''
