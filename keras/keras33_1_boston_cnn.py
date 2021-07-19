import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7, shuffle=True)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성 # ??
model = Sequential() 
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(13))) 
model.add(Conv2D(20, (2,2), activation='relu')) 
model.add(Conv2D(30, (2,2), padding='valid')) 
model.add(MaxPooling2D())
model.add(Conv2D(15, (2,2))) 
model.add(Flatten()) 
model.add(Dense(64,activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일 및 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=512, verbose=2,
    validation_split=0.0005, callbacks=[es])

# 4. predict eval 

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])
