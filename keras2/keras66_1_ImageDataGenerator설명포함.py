import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# randomsearch보다 빠르고 파라미터도 자동으로 부여해준다 

# ================ ImageDataGenerator정의 ====================
# train ---> 학습필수
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
) # rotation_range 반복값
# zoom_range확대, fill_mode 빈자리를 주변값으로 채운다(padding도 있다(빈자리0으로))

# test --> 전처리만
test_datagen = ImageDataGenerator(rescale=1/255)

#  flow또는 flow_from_direstory - 폴더에 있는파일을 그대로 가져온다
# label -> y
# ===================== generator ===========================
# train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/train',
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
) # fit과 같은 기능, 실제 이미지파일 전까지 디렉토리 지정
# print(x.shape, y.shape) #(80,150,150,1), (80,)

'''
target_size가로세로 150으로 만들예정
x.shape(80,150,150,1)->증폭 아직안됨, y.shape(80,) --> 값 rescale=1/255이므로 x는 0~1까지 값과 y는 전부다 0,1이다
이진분류이므로 y의 값은 0,1로만 있고
x값은 원래 0~255인데 rescale/255이므로 0~1로 구성이 바뀐다
'''
# test
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

print(xy_train) # 2 classes(ad, normal)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000020974E58550> -> 딕셔너리형태
print(xy_train[0]) # -> x, array=4차원(x도 5개)
# [0., 1., 1., 0., 0.] -> y
print(xy_train[0][0]) # -> x
print(xy_train[0][0].shape) # -> x (5, 150, 150, 3)왜 5개만 나옴 80개인데? xy_train배치사이즈가 5이므로
print(xy_train[0][1]) # -> y [0. 1. 0. 1. 1.]
print(xy_train[0][1].shape) # -> y (5,) ---> xy_train[0] 0은 160장의 데이터 이므로 0~15까지 넣을 수 있다
print(xy_train[15][1].shape) # -> y (5,) ---> batch_size=10이면 xy_train[0] 0은 160장의 데이터 이므로 0~15까지 넣을 수 있다
# batch_size=10이면 xy_train[0][0]은 앞에 0은 160장이므로 0~15, 뒤에 0은 (x,y)이므로 0,1을 쓸 수 있다
# batch_size = 160까지 지정가능(더 크게 해도 160까지만 나옴)
