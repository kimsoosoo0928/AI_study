import numpy as np
import pandas as pd # 판다스로 csv를 땡겨오는게 편하다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';'
                        ,index_col=None, header=0) # ;을 기준으로 데이터를 분리한다.

# ./ : 현재폴더
# ../ : 상위폴더

print(datasets)
print(datasets.shape) #(4898, 12)
# x = (4898,11) 
# y = (4898,) : quality
print(datasets.info())
print(datasets.describe())


# 다중분류
# 모델링하고
# 0.8 이상 완성!!

#1. 판다스 -> 넘파이
#2. x와 y를 분리
#3. sklearn의 onehot??? 사용할것 
#4. y의 라벨을 확인 np.unique(y)
#5. y의 shape 확인 (4898,) -> (4898,7)
# 1-1 데이터

num_datasets = datasets.to_numpy()
x = num_datasets[:,0:11]
y = num_datasets[:,11:]

encoder = LabelEncoder()
encoder.fit(y)
labels = encoder.transform(y)
# 2차원 데이터로 변환합니다. 
labels = labels.reshape(-1,1)
# 원-핫 인코딩을 적용합니다. 
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape) # 0,1,2가 생략된다.

# 컨닝!

