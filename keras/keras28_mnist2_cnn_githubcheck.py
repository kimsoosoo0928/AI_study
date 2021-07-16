import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 60000장
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) 10000징

# 데이터 전처리

x_train = x_train.reshape(60000, 28, 28, 1) # 3차원데이터 => 4차원데이터
x_test = x_test.reshape(10000, 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0
print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 만들어보기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Flatten(input_shape=(28,28))) 
model.add(Dense(128,activation='relu'))
model.add(Dense(64))
model.add(Dense(1, activation='softmax'))

model.summary()
#3. 컴파일, 훈련 metrics=['acc']

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

model.fit(x_train, y_train, epochs=5, batch_size=8,)

#4. 평가, 예측 predict할 필요는 없다.
loss, acc = model.evaluate(x_test, y_test) 
print('accuracy : ', acc)

# acc로만 판단해보자!!