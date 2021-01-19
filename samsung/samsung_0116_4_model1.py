## import 코드
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, concatenate, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

## npz 파일 불러오기
datasets=np.load('C:/data/npy/samsung_day_0115.npz')
datasets2=np.load('C:/data/npy/kodex_day_0115.npz')

x1_train=datasets['x_train']
x1_test=datasets['x_test']
x1_val=datasets['x_val']
x1_pred=datasets['x_pred']

x2_train=datasets2['x_1_train']
x2_test=datasets2['x_1_test']
x2_val=datasets2['x_1_val']
x2_pred=datasets2['x_1_pred']

y_train=datasets['y_train']
y_test=datasets['y_test']
y_val=datasets['y_val']

model = load_model('C:/data/h5/samsung0115_3_model.h5')

##4. 평가, 예측
loss=model.evaluate([x1_test, x2_test], y_test)
y_pred=model.predict([x1_test, x2_test])

pred=model.predict([x1_pred, x2_pred])

## 출력
print('loss:', loss)
print(pred)

'''
[[90478.85 90301.4]]
'''
