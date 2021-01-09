####CNN 정의
##### https://keras.io/api/layers/convolution_layers/convolution2d/
#위 사이트 참조
'''
컨볼루션 자체는 4차원
Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), 
                                          activation='relu')

주요 인자는 다음과 같습니다.

첫번째 인자 : (filters)컨볼루션 필터의 수 입니다. (=노드의 수)
두번째 인자 : (커널)컨볼루션 커널의 (행, 열) 입니다. (가로, 세로 몇개로 자를지)
padding : 경계 처리 방법을 정의합니다.
‘valid’ : 유효한 영역만 출력이 됩니다. 
          따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
strides : 연산을 수행할 때 윈도우가 가로 그리고 세로로 움직이면서 
          내적 연산을 수행하는데, 한 번에 얼마나 움직일지를 의미한다.
          디폴트 값은 1
MaxPooling2D : 중요하지 않은 것 날림        
input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 
              모델에서 첫 레이어일 때만 정의하면 됩니다.
              (전체행, 전체열, 색(=채널))로 정의합니다. 
              (rows, cols, channels) ->확대시(batch,rows, cols, channels)
              흑백영상인 경우에는 채널이 1이고,
              컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
activation : 활성화 함수 설정합니다.
‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 
           그대로 출력으로 나옵니다.
‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
‘softmax’ : 소프트맥스 함수, 
            다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
입력 형태는 다음과 같습니다.

image_data_format이 ‘channels_first’인 경우 
(샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서입니다.
image_data_format이 ‘channels_last’인 경우 
(샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서입니다.

image_data_format 옵션은 “keras.json” 파일 안에 있는 설정입니다. 
콘솔에서 “vi ~/.keras/keras.json”으로 
keras.json 파일 내용을 변경할 수 있습니다.

출력 형태는 다음과 같습니다.

image_data_format이 ‘channels_first’인 경우 
(샘플 수, 필터 수, 행, 열)로 이루어진 4D 텐서입니다.
image_data_format이 ‘channels_last’인 경우 
(샘플 수, 행, 열, 필터 수)로 이루어진 4D 텐서입니다.

행과 열의 크기는 padding가 ‘same’인 경우에는 
입력 형태의 행과 열의 크기가 동일합니다.

간단한 예제로 컨볼루션 레이어와 필터에 대해서 알아보겠습니다.
입력 이미지는 채널 수가 1, 너비가 3 픽셀, 높이가 3 픽셀이고, 
크기가 2 x 2인 필터가 하나인 경우를 레이어로 표시하면 다음과 같습니다. 
단 image_data_format이 ‘channels_last’인 경우 입니다.

conv2D(1, (2,2), input_shape=(3,3,1))
'''

