# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.


# https://gist.github.com/SphericalKat/7d67d57f784db085a6b734f4929aa004
import csv
import tensorflow as tf
import numpy as np
import urllib
# 태양 흑점 파악 - 시계열 문제(split함수)
# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    '''
    window_dataset에 대해
    # series = tf.expand_dims(series, axis=-1)
    expand_dims 하기 전
    익스팬드 전 [0.24284279 0.26192868 0.29306881 ... 0.03314917 0.03992968 0.00401808]
    shape=(3235,)

    그러나!!!
    expand적용 후
    [[0.24284279]
    [0.26192868]
    [0.29306881]
    ...
    [0.03314917]
    [0.03992968]
    [0.00401808]], 
    shape=(3235, 1), dtype=float64)
    리세이프 개념과 비슷
    '''
    ds = tf.data.Dataset.from_tensor_slices(series)
    # from_tensor_slices : np.array를 tf.Dataset으로
    # dtype=float64
    # (3235,1)짜리를 > (1,)*3235 개로 한 줄 씩 나눔
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # window _ shift씩 이동해 (window_size(x) + 1(y))만큼 그룹화한다, drop_remainder=True남은부분 버린다
    # 한 칸씩 띄어서 31개씩 잘라서 반복하여 데이터로 만든다
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # flat_map : lambda 함수를 불러 (window_size(x) + 1(y))만큼 읽은 후 단일 dataset으로 반환
    # 세트를 1차원으로 하나로 만듬
    ds = ds.shuffle(shuffle_buffer)
    # shuffle: shuffle_buffer를 랜덤으로 추출
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    # map : 함수를 맵핑, w[:-1] -> x, w[1:] -> y
    return ds.batch(batch_size).prefetch(1)
    # Data prefetch란 앞으로 연산에 필요한 data들을 미리 가져오는 것
    # 병렬처리로 훈련 속도 향상

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    series = np.array(sunspots)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function(정규화 함수)
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max # 전처리
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size,
                         batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    vaild_set = windowed_dataset(x_valid, window_size=window_size,
                         batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(40, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),      
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
      tf.keras.layers.Dense(1)
    ])

    # ================== 훈련, 컴파일 ====================
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    model.fit(train_set, batch_size=128, epochs=100)

    # ================== 예측 ======================
    acc = model.evaluate(vaild_set)
    print('acc', acc[1])

    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # 훈련하는 동안이 텍스트를 본다면, 무시해도 안전합니다.
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #


    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    return model

'''
acc 0.004878048785030842 
정확도 무선129...
'''
# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Study/tf_certificate/Category5/mymodel.h5")


# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
#def model_forecast(model, series, window_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(32).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast


#window_size = # YOUR CODE HERE
#rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
#rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

#result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

## To get the maximum score, your model must have an MAE OF .12 or less.
## When you Submit and Test your model, the grading infrastructure
## converts the MAE of your model to a score from 0 to 5 as follows:

#test_val = 100 * result
#score = math.ceil(17 - test_val)
#if score > 5:
#    score = 5

#print(score)