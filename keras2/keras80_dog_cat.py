# 이미지는 
## data/image/vgg/ 에 4개 넣으시오
# 개, 고양이, 라이언, 수트
# 넣어놩
# 파일명
# dog1.jpg cat1.jpg lion.jpg suit1.jpg

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224, 224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224, 224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size=(224, 224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224, 224))

# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog) # 이미지 array로 바꿈 -> 수치화
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(arr_dog)
# print(type(arr_dog)) # <class 'numpy.ndarray'>
# print(arr_dog.shape) # (224, 224, 3)

# RGB -> BGR 
from tensorflow.keras.applications.vgg16 import preprocess_input # 알아서 vgg16 맞춰 preprocess_input 해준다
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
# print(arr_dog.shape) # (224, 224, 3)
# print(arr_dog)

# 위 arr 4개 이미지를 3차원을 합쳐준다. (224, 224, 3)가 네개 이므로 -> (4, 224, 224, 3)
arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit]) # stack을 사용해 이어 붙이기
print(arr_input.shape)

# 2. 모델구성
model = VGG16()
results = model.predict(arr_input)

print(results)
print('results.shape : ', results.shape) # 이미지에 분류할 수 있는 카테고리 수 :1000개

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions # 예측해석 코드(결과)

decode_results = decode_predictions(results)
print('===========================================')
print('result[0] : ', decode_results[0])
print('===========================================')
print('result[1] : ', decode_results[1])
print('===========================================')
print('result[2] : ', decode_results[2])
print('===========================================')
print('result[3] : ', decode_results[3])
print('===========================================')


