import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

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

# test --> 전처리만
test_datagen = ImageDataGenerator(rescale=1/255)

# ===================== generator ===========================
# train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/train/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    save_to_dir='C:/data/image/brain/brain_generator/train/'
) # fit과 같은 기능, 실제 이미지파일 전까지 디렉토리 지정
# print(x.shape, y.shape) #(80,150,150,1), (80,)

# test
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    save_to_dir='C:/data/image/brain/brain_generator/teat/'
)

print(xy_train[0][0]) # 이렇게 정의해야 데이터 폴더에 저장
print(xy_train[0][1]) # 이렇게 정의해야 데이터 폴더에 저장
# print개수 * batch_size 수만큼 폴더에 파일 생성
#((print문 2개 batch 6개면 12개 사진 생긴다))
# batch_size 160이상 이어도 print문 * 160 이다