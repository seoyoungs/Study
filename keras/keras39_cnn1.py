####앞에 keras39_cnn.py 설명보기


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D

model =Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(10, 10,1)))
##여기서 넘어가면 (9,9,10) ---> 실직적으로는 4차원(None, 9,9,10)
model.add(MaxPooling2D(pool_size=(3,3), padding='same')) #시작점에는 못쓴다. model.add(Conv2D)정의 한 다음에 쓸 수 있다.
model.add(Conv2D(9, (2,2), padding='valid'))
#model.add(Conv2D(9, (2,3)))
#model.add(Conv2D(8, 2))  #이렇게 연속해서 Conv2D써도 된다
####하지만 특성이 좋아질까?? LSTM과 다르게 특성을 추출하는 기능이다.
##그래서 추출을 위해 계속 쓰는 것이 좋아질 수 있다.
model.add(Flatten())
model.add(Dense(1))
model.summary()


'''
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(10, 10,1)))
model.add(Dense(1))
이렇게 하면 2차원이 나온다. 1차원을 만들어야 정확히 이미지 식별이 가능하다.

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
dense (Dense)                (None, 9, 9, 1)           11
=================================================================

따라서 2차원으로 받아들일 수 있는 함수 필요함
 Flatten을 추가해 2차원으로 받아들이게 해야함 Conv2D의 특징 때문에 
 그대로 4차원으로 받아들임

LSTM과 다른 점

###param 계산 방법 
# padding적용전 (디폴트 valid)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           296
_________________________________________________________________
flatten (Flatten)            (None, 392)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 393
=================================================================
2x2(필터 크기) x 1(#입력 채널(RGB))x 10(#출력 채널) +10(출력 채널 bias)=50
2x2(필터 크기) x 10(#입력 채널(RGB))x 9(#출력 채널) +9(출력 채널 bias)=369
(input_dim * kernel_size + bias) * filter(=channel) = (1*2*2+1)*10 =50

# padding적용 후 padding='same'
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________

디폴트 valid= (None, 9, 9, 10)  
padding='same' (None, 10, 10, 10)
'same'적용 시 input_shape=(10, 10,1)이 그대로 유지된 것을 알 수 있다.

####strides
strides : 연산을 수행할 때 윈도우가 가로 그리고 세로로 움직이면서 
          내적 연산을 수행하는데, 한 번에 얼마나 움직일지를 의미한다.
=================================================================
conv2d (Conv2D)              (None, 5, 5, 10)          50
_________________________________________________________________

이렇게 (None, 10, 10, 10) 에서 두칸씩 이동하므로 한 번에 5,5로 변형된다.
디폴트 값은 1이기 때문에 그동안 입력하지 않으면 10, 10 인것이다

model.add(MaxPooling2D(pool_size=2))
디폴트 값 2
max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0

model.add(MaxPooling2D(pool_size=3))
max_pooling2d (MaxPooling2D) (None, 3, 3, 10)          0
세개씩 묶어 하나 날리는 것 (10,10)이니 하나가 남는다.

model.add(MaxPooling2D(pool_size=3))
max_pooling2d (MaxPooling2D) (None, 4, 4, 10)          0
이렇게 하면 하나가 안날라 간다.

model.add(MaxPooling2D(pool_size=(2,3)))
max_pooling2d (MaxPooling2D) (None, 5, 3, 10)          0

'''


