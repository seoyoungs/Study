import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#보스톤 데이터랑 가타. 불러오는게 (tensorflow)

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#다중분류이므로 원핫코딩, 전처리 해야한다.
##이걸로 전처리 해서 MinMaxscaler안해도 된다.
x_train= x_train.reshape(60000, 28,28, 1).astype('float32')/255.
x_test= x_test.reshape(10000, 28,28, 1)/255.
#이미지 특성맞춰 숫자 바꾸기 x의 최대가 255이므로 255로 나눈다.
#이렇게 하면 0~1 사이로 된다.
#x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
#x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)

#다중분류 y원핫코딩
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  #(10000, 10)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model =Sequential()
model.add(Conv2D(filters=50, kernel_size=(3,3), 
                padding='same', strides=(1,1), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='softmax')) #y값 3개이다(0,1,2)
#model.summary()

#ModelCheckpoint
import datetime
# date_now = datetime.datetime.now() # 문제점
# print(date_now) # 주의점 컴퓨터내 시간제공시간과 같다->2021-01-27 10:07:14.297268
# date_time = date_now.strftime("%m%d_%H%M")
# print(date_time) # 주의점 컴퓨터내 시간제공시간과 같다->0127_1013(월일_시간분)
# 컴퓨터에 지정된 현재 시간이 출력 됨(첫 시작 시각만 출력)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath='../data/modelCheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k45_", '{timer}', filename])

#------------------요거 실시간저장 추가--------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M%S'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath
#--------------------------------------------------------------------------------

#modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#02d 정수로 두번째 자리 까지, 4f 실수로 4번째 자리까지
#따라서 0.01이면 02d: 01, 4f : 0100이된다. k45_mnist_0100.hdf5

# 컴파일, 훈련
cp = MyModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                       save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
#ModelCheckpoint는 최저점이 생길 때마다 filepath(파일형태)로 기록한다.
#파일형태에 weight 값을 기록하기 위해 사용한다. 최적의 weight(val_loss가장낮음)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
####loss가 이진 분류일 때는binary_crossentropy(0,1만 추출)
hist = model.fit(x_train, y_train, epochs=15, batch_size=32, 
                 validation_split=0.2, callbacks=[es, cp])

#4. 평가 훈련
result=model.evaluate(x_test,y_test, batch_size=16)
print('loss : ', result[0])
print('acc : ', result[1])
'''
loss :  0.129032164812088
acc :  0.9627000093460083
'''

# y_pred = model.predict(x_test[:10])
# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))

'''
####시각화
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.figure(figsize=(10,6)) #가로세로 #이거 한 번 만 쓰기
plt.subplot(2,1,1)   #서브면 2,1,1이면 2행 1열 중에 1번째라는 뜻
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() #바탕을 grid모눈종이로 하겠다

#plt.title('손실비용') #한글깨짐 오류 해결할 것 과제1
plt.title('Cost loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

import matplotlib.pyplot as plt
plt.subplot(2,1,2)   #2행 1열 중에 2번째라는 뜻
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()
'''
