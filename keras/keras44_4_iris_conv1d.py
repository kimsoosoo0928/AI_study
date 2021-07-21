# diabets
# LSTM

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 
datasets = load_iris()

x = datasets.data # (150, 4) 
y = datasets.target # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

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
model.add(Conv1D(64, 2, input_shape=(4, 1)))
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
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
time :  15.791856527328491
loss :  0.06549330800771713
R^2 score :  0.9011007137945044

LSTM + Conv1D
time :  10.505834817886353
loss :  0.05814383924007416
R^2 score :  0.9121988969431949
'''