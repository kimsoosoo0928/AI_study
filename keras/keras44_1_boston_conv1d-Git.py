# boston
# LSTM

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Flatten

# 1. data 
datasets = load_boston()

x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

size = 6

def split_x(dataset, size):
    aaa = [] 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i + size)]
        aaa.append(subset) 
    return np.array(aaa) 

dataset = split_x(x, size)

print(dataset)

x = dataset[:, :5].reshape(13, 1)
y = dataset[:, 5]

print(x.shape)

print("x : \n", x)
print("y : ", y)


# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D

model = Sequential()
# model.add(LSTM(64, input_shape=(5, 1)))
model.add(Conv1D(64, 2, input_shape=(5, 1)))
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
model.fit(x, y, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
y_predict = model.predict([x])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x,y)
print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y, y_predict)
print('R^2 score : ', r2)

'''
LSTM
time :  87.23254299163818
loss :  13.674508094787598
R^2 score :  0.8363959289784044



'''

