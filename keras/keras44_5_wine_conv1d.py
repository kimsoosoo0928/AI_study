import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 )

x = datasets.iloc[:,0:11] # (4898, 11)
y = datasets.iloc[:,[11]] # (4898, 10)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y)
y = one.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.005, shuffle=True, random_state=24)

# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(11, 1)))
model.add(LSTM(64, return_sequences=True)) # LSTM 다음에 Conv 사용하는경우가 많다.
model.add(Conv1D(64,2 ))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 3. compile, fit
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
y_predict = model.predict([x_test])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])