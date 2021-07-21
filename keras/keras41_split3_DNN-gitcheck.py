#keras41_split2_LSTM을 DNN으로 바꾸시오.
#R2와 RMSE 비교해볼것

# 1 ~ 100 까지의 데이터를 

# x                    y
# 1, 2, 3, 4, 5        6
# ...
# 95, 96, 97, 98, 99   100

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. data

x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))
#         x               y
# 96, 97, 98, 99, 100     ?
# ...
# 101, 102, 103, 104, 105 ?

# 예상 결과값 : 101, 102, 103, 104, 105, 106

size = 5

def split_x(dataset, size):
    aaa = [] 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i + size)]
        aaa.append(subset) 
    return np.array(aaa) 

dataset = split_x(x_data, size)

print(dataset)

x = dataset[:, :4]
y = dataset[:, 4]

print("x : \n", x)
print("y : ", y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)


# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape)
print(x_test.shape)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
model = Sequential()
model.add(Dense(1024, input_shape =(4,1), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

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